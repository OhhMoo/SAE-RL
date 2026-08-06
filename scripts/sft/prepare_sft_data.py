"""Build verl-SFT-format GSM8k data from the existing PPO parquet.

Reads data/gsm8k/{train,test}.parquet (verl PPO format) and writes
data/gsm8k_sft/{train,test}.parquet with a single `messages` column that
verl's MultiTurnSFTDataset consumes (messages_key=messages).

Each row's messages = the existing [system, user] prompt turns (verbatim, so
the SFT model sees the same prompt/format the PPO chains use) + an assistant
turn whose content is the full worked GSM8k solution (ends with '#### N').
"""

import os

import pandas as pd
from datasets import Dataset

SRC_DIR = "data/gsm8k"
OUT_DIR = "data/gsm8k_sft"


def to_messages(row) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in row["prompt"]]
    msgs.append({"role": "assistant", "content": row["extra_info"]["answer"]})
    return msgs


def build(split: str) -> None:
    df = pd.read_parquet(os.path.join(SRC_DIR, f"{split}.parquet"))
    records = [{"messages": to_messages(r)} for _, r in df.iterrows()]
    ds = Dataset.from_list(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{split}.parquet")
    ds.to_parquet(out)

    ex = records[0]["messages"]
    print(f"[{split}] {len(ds)} rows -> {out}")
    print(f"[{split}] roles: {[m['role'] for m in ex]}")
    print(f"[{split}] assistant tail: {ex[-1]['content'][-40:]!r}")
    assert [m["role"] for m in ex] == ["system", "user", "assistant"], "unexpected role layout"
    assert "####" in ex[-1]["content"], "assistant content missing '#### N' answer"


if __name__ == "__main__":
    for split in ("train", "test"):
        build(split)
