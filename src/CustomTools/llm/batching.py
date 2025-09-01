import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from IPython.display import clear_output
from openai import OpenAI
import anthropic

gpt_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o4-mini"];
claude_models = ["claude-3-7-sonnet", "claude-3-5-haiku"];
models = gpt_models + claude_models;

_default_costs = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o4-mini": (1.10, 4.40),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
};

def set_cost(model: str, input_per_million: float, output_per_million: float) -> None:
    _default_costs[model] = (input_per_million, output_per_million);

def get_cost(model: str) -> Tuple[float, float]:
    if model not in _default_costs:
        raise ValueError(f"Unknown model: {model}");
    ic, oc = _default_costs[model];
    return ic / 1_000_000, oc / 1_000_000;

def list_models() -> None:
    print("Available models:");
    for m in models:
        ic, oc = get_cost(m);
        print(f"{m}, input cost: ${ic*1_000_000}/M, output cost: ${oc*1_000_000}/M.");

def _get_keys() -> Tuple[Optional[str], Optional[str]]:
    return os.environ.get("OPENAI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY");

def CallLLM(assistant_prompt: str, user_prompt: str, labels: Optional[List[str]] = None, images: Optional[List[str]] = None, model: str = "gpt-5", max_tokens: int = 2000, temp: float = 0.0, force_json: bool = False) -> Tuple[str, Dict[str, Any]]:
    ic, oc = get_cost(model);
    if labels is not None:
        lbl = "; ".join(labels) if isinstance(labels, list) else str(labels);
        assistant_prompt = assistant_prompt.replace("<\\labels\\>", lbl);
    openai_key, anthropic_key = _get_keys();
    if model in gpt_models:
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing");
        client = OpenAI(api_key=openai_key);
        messages: List[Dict[str, Any]] = [{"role": "system", "content": assistant_prompt}];
        user_content: List[Dict[str, Any]] = [];
        if images:
            for url in images:
                user_content.append({"type": "image_url", "image_url": {"url": url}});
        user_content.append({"type": "text", "text": user_prompt});
        messages.append({"role": "user", "content": user_content});
        if model in ["gpt-5", "gpt-5-mini", "gpt-5-nano"]:
            params = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "temperature": temp or 0.0};
            if force_json:
                params["response_format"] = {"type": "json_object"};
        else:
            params = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp};
            if force_json:
                params["response_format"] = {"type": "json_object"};
        resp = client.chat.completions.create(**params);
        raw = resp.choices[0].message.content or "";
        content = json.loads(raw) if force_json else raw;
        input_tokens = int(resp.usage.prompt_tokens or 0);
        output_tokens = int(resp.usage.completion_tokens or 0);
    elif model in claude_models:
        if not anthropic_key:
            raise RuntimeError("ANTHROPIC_API_KEY missing");
        client = anthropic.Anthropic(api_key=anthropic_key);
        content_parts: List[Dict[str, Any]] = [];
        if images:
            for url in images:
                content_parts.append({"type": "image", "source": {"type": "url", "url": url}});
        content_parts.append({"type": "text", "text": user_prompt});
        sys = assistant_prompt if not force_json else assistant_prompt + "\nRespond with valid JSON only.";
        resp = client.messages.create(model=model, system=sys, messages=[{"role": "user", "content": content_parts}], max_tokens=max_tokens, temperature=temp);
        raw = "".join([c.text for c in resp.content if c.type == "text"]);
        content = json.loads(raw) if force_json else raw;
        input_tokens = int(resp.usage.input_tokens or 0);
        output_tokens = int(resp.usage.output_tokens or 0);
    else:
        raise ValueError(f"Unsupported model: {model}");
    total_cost = ic * input_tokens + oc * output_tokens;
    return assistant_prompt + user_prompt, {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens, "input_cost": f"${ic*input_tokens:.4f}", "output_cost": f"${oc*output_tokens:.4f}", "total_cost": f"${total_cost:.4f}"};

def BatchUploader(df, id_col: str, inf_col: str, role: str, question: str, label_col: Optional[str] = None, image_col: Optional[str] = None, model: str = "gpt-4o", max_tokens: int = 1000, temp: float = 0.0, force_json: bool = False, description: str = "", path: Optional[str] = None, upload: bool = False, estimate: bool = True) -> List[Any]:
    ic, oc = get_cost(model);
    base_dir = path or "./Batches";
    os.makedirs(base_dir, exist_ok=True);
    input_path = os.path.join(base_dir, "requests.jsonl");
    if estimate and len(df) > 0:
        ex = df.iloc[0];
        rc = role.replace("<\\labels\\>", str(ex[label_col])) if label_col else role;
        ex_prompt = question + str(ex[inf_col]);
        _, info = CallLLM(assistant_prompt=rc, user_prompt=ex_prompt, images=None, model=model, max_tokens=max_tokens, temp=temp, force_json=force_json);
        est = info["input_tokens"] * ic * len(df) + info["output_tokens"] * oc * len(df);
        print(f"Estimated cost: ${est:.3f}");
    with open(input_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rc = role.replace("<\\labels\\>", str(row[label_col])) if label_col else role;
            messages: List[Dict[str, Any]] = [{"role": "system", "content": rc}];
            if image_col and row.get(image_col):
                urls = row[image_col] if isinstance(row[image_col], list) else json.loads(row[image_col]);
                user_content: List[Dict[str, Any]] = [{"type": "image_url", "image_url": {"url": u}} for u in urls];
                user_content.append({"type": "text", "text": question + str(row[inf_col])});
                messages.append({"role": "user", "content": user_content});
            else:
                messages.append({"role": "user", "content": question + str(row[inf_col])});
            body: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temp};
            if force_json:
                body["response_format"] = {"type": "json_object"};
            data = {"custom_id": str(row[id_col]), "method": "POST", "url": "/v1/chat/completions", "body": body};
            f.write(json.dumps(data, ensure_ascii=False) + "\n");
    max_size = 199 * 1024 * 1024;
    max_rows = 50000;
    file_count = 1;
    cur_size = 0;
    row_count = 0;
    out_path = os.path.join(base_dir, f"Batch{file_count}.jsonl");
    out_f = open(out_path, "w", encoding="utf-8");
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            size = len(line.encode("utf-8"));
            if cur_size + size > max_size or row_count >= max_rows:
                out_f.close();
                file_count += 1;
                out_path = os.path.join(base_dir, f"Batch{file_count}.jsonl");
                out_f = open(out_path, "w", encoding="utf-8");
                cur_size = 0;
                row_count = 0;
            out_f.write(line);
            cur_size += size;
            row_count += 1;
    out_f.close();
    if not upload:
        print(f"Batches created at {base_dir}.");
        return [];
    openai_key, anthropic_key = _get_keys();
    all_batches: List[Any] = [];
    if model in gpt_models:
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing");
        client = OpenAI(api_key=openai_key);
        for i in range(1, file_count + 1):
            batch_file = open(os.path.join(base_dir, f"Batch{i}.jsonl"), "rb");
            file_obj = client.files.create(file=batch_file, purpose="batch");
            batch = client.batches.create(input_file_id=file_obj.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": f"{description} (Batch {i})"});
            all_batches.append(batch);
            print(f"Batch {i} -> ID {batch.id}");
    elif model in claude_models:
        raise NotImplementedError("Batch upload for Anthropic is not implemented in this module");
    else:
        raise ValueError(f"Unsupported model: {model}");
    return all_batches;

def BatchChecker(batch_id: Any, model: str, output_dir: str = "./Output", poll_interval: float = 1.0) -> None:
    if not isinstance(batch_id, list):
        batch_id = [batch_id];
    os.makedirs(output_dir, exist_ok=True);
    status_map: Dict[str, str] = {b: "pending" for b in batch_id};
    done: List[str] = [];
    if model in gpt_models:
        openai_key, _ = _get_keys();
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing");
        client = OpenAI(api_key=openai_key);
        while True:
            for b in batch_id:
                if b in done:
                    continue;
                retrieved = client.batches.retrieve(b);
                status_map[b] = f"{retrieved.status} / {retrieved.request_counts}";
                if retrieved.status in {"completed", "expired"}:
                    done.append(b);
                    fr = client.files.content(retrieved.output_file_id);
                    with open(os.path.join(output_dir, f"output_{b}.txt"), "w", encoding="utf-8") as outfile:
                        outfile.write(fr.text);
                elif retrieved.status == "failed":
                    done.append(b);
                time.sleep(poll_interval);
            clear_output();
            for b in batch_id:
                print(f"{b}: {status_map[b]}");
            if len(done) >= len(status_map):
                break;
        print("All ready!");
    elif model in claude_models:
        raise NotImplementedError("Batch checking for Anthropic is not implemented in this module");
    else:
        raise ValueError(f"Unsupported model: {model}");

def BatchRetriever(df, id_col: str, folder_path: str = "./Output") -> Any:
    data_map: Dict[str, Any] = {};
    bad_lines = 0;
    tqdm.pandas(desc="Bringing outputs...");
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                for row in file:
                    line = row.strip().replace("```json", "").replace("```", "").strip();
                    try:
                        data = json.loads(line);
                        custom_id = str(data["custom_id"]);
                        content = data["response"]["body"]["choices"][0]["message"]["content"];
                        data_map[custom_id] = content;
                    except (json.JSONDecodeError, KeyError, TypeError):
                        bad_lines += 1;
    df["response"] = df[id_col].astype(str).map(data_map);
    print(f"Process completed. {bad_lines} lines could not be processed.");
    return df;