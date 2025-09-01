import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found in environment.");

client = OpenAI();

def embed(df, id="id", text="text", model="text-embedding-3-large"):
    descs = df[text].tolist();
    embeddings = [];
    for i in tqdm(range(0, len(descs), 64), desc="Embedding texts..."):
        batch = descs[i:i + 64];
        resp = client.embeddings.create(model=model, input=batch);
        embeddings.extend([d.embedding for d in resp.data]);

    if len(embeddings) != len(descs):
        raise RuntimeError("Embedding mismtach.");

    dim = len(embeddings[0]);
    embed_df = pd.DataFrame(embeddings, columns=[f"d{i}" for i in range(dim)]);
    df_emb = pd.concat([df.reset_index(drop=True), embed_df], axis=1);
    return df_emb