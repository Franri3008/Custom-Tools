import json
from typing import Optional, List, Dict, Any, Tuple
import os
from openai import OpenAI
import anthropic

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def single_call(assistant_prompt: str, user_prompt: str, labels: Optional[List[str]] = None, images: Optional[List[str]] = None, model: str = "gpt-5", max_tokens: int = 2000, temp: float = 0.0, force_json: bool = False) -> Tuple[str, Dict[str, Any]]:
    gpt_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o4-mini"];
    claude_models = ["claude-3-7-sonnet", "claude-3-5-haiku"];

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

    def get_cost(model: str) -> Tuple[float, float]:
        if model not in _default_costs:
            raise ValueError(f"Unknown model: {model}");
        ic, oc = _default_costs[model];
        return ic / 1_000_000, oc / 1_000_000;

    def _get_keys() -> Tuple[Optional[str], Optional[str]]:
        return os.environ.get("OPENAI_API_KEY"), os.environ.get("ANTHROPIC_API_KEY"); 
    
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


def multi_call(df: pd.DataFrame, max_workers: int=10, labels: Optional[List[str]]=None, images: Optional[List[str]]=None, model: str = "gpt-5", max_tokens: int = 2000, temp: float = 0.0, force_json: bool = False) -> pd.DataFrame:
    if "system_prompt" not in df.columns or "user_prompt" not in df.columns:
        raise ValueError("DataFrame must contain 'system_prompt' and 'user_prompt' columns");
    out = df.copy();
    out["response"] = None;
    futures = {};
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in out.iterrows():
            futures[executor.submit(single_call, str(row["system_prompt"]), str(row["user_prompt"]), labels, images, model, max_tokens, temp, force_json)] = idx;
        pbar = tqdm(total=len(futures));
        for fut in as_completed(futures):
            idx = futures[fut];
            try:
                _, meta = fut.result();
                out.at[idx, "response"] = meta.get("content", "");
            except Exception as e:
                out.at[idx, "response"] = str(e);
            pbar.update(1);
        pbar.close();
    return out