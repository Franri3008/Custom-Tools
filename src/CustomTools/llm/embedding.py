import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

def embed(df:pd.DataFrame, model="text-embedding-3-large"):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment.");
    client = OpenAI();

    descs = df["text"].tolist();
    embeddings = [];
    for i in tqdm(range(0, len(descs), 64), desc="Embedding texts..."):
        batch = descs[i:i + 64];
        resp = client.embeddings.create(model=model, input=batch);
        embeddings.extend([d.embedding for d in resp.data]);

    if len(embeddings) != len(descs):
        raise RuntimeError("Embedding mismatch.");

    dim = len(embeddings[0]);
    embed_df = pd.DataFrame(embeddings, columns=[f"d{i}" for i in range(dim)]);
    df_emb = pd.concat([df.reset_index(drop=True), embed_df], axis=1);
    return df_emb

def cosine_similarity(df1:pd.DataFrame, df2:pd.DataFrame, descriptions=False):
    if descriptions and ("text" not in df1.columns or "text" not in df2.columns):
        raise RuntimeError("'text' column not found.");
    emb1_cols = [c for c in df1.columns if c.startswith("d") and c[1:].isdigit()];
    emb2_cols = [c for c in df2.columns if c.startswith("d") and c[1:].isdigit()];
    X1 = df1[emb1_cols].to_numpy(dtype=np.float64);
    X2 = df2[emb2_cols].to_numpy(dtype=np.float64);
    norms1 = np.linalg.norm(X1, axis=1, keepdims=True);
    norms2 = np.linalg.norm(X2, axis=1, keepdims=True);
    X1 = X1 / norms1;
    X2 = X2 / norms2;
    S = X1 @ X2.T;
    long = pd.DataFrame(S).stack().reset_index().rename(columns={0: "similarity"});
    long.columns = ["i", "j", "similarity"];
    i = long["i"].to_numpy();
    j = long["j"].to_numpy();
    if descriptions:
        res = pd.DataFrame({
            "from": df1.loc[i, "id"].to_numpy(),
            "from_desc": df1.loc[i, "text"].to_numpy(),
            "to": df2.loc[j, "id"].to_numpy(),
            "to_desc": df2.loc[j, "text"].to_numpy(),
            "similarity": long["similarity"].to_numpy()
        })
    else:
        res = pd.DataFrame({
            "from": df1.loc[i, "id"].to_numpy(),
            "to": df2.loc[j, "id"].to_numpy(),
            "similarity": long["similarity"].to_numpy()
        })
    return res