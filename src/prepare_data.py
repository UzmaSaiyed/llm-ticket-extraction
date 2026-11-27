import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.schema import INTENT_TO_CATEGORY

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUT_DIR = Path("data")
SEED = 42
PER_INTENT = {"train": 111, "validation": 18, "test": 18}


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s{}]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_clean() -> pd.DataFrame:
    df = load_dataset(DATASET, split="train").to_pandas()
    df = df[["instruction", "category", "intent"]].rename(columns={"instruction": "text"})
    df["text"] = df["text"].str.strip()
    df["key"] = df["text"].map(normalize)

    # same text with different intents would be an unfair question, so drop it
    intents_per_key = df.groupby("key")["intent"].nunique()
    df = df[~df["key"].isin(intents_per_key[intents_per_key > 1].index)]
    df = df.drop_duplicates(subset="key")

    assert (df["intent"].map(INTENT_TO_CATEGORY) == df["category"]).all()
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {name: [] for name in PER_INTENT}
    for _, group in df.groupby("intent"):
        group = group.sample(frac=1, random_state=SEED)
        start = 0
        for name, count in PER_INTENT.items():
            parts[name].append(group.iloc[start:start + count])
            start += count
    return {
        name: pd.concat(chunks).sample(frac=1, random_state=SEED).reset_index(drop=True)
        for name, chunks in parts.items()
    }


def main() -> None:
    raw_rows = len(load_dataset(DATASET, split="train"))
    df = load_clean()
    print(f"rows: {raw_rows} -> {len(df)} after removing duplicates")

    splits = split(df)
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a]["key"]) & set(splits[b]["key"]), f"{a}/{b} overlap"

    OUT_DIR.mkdir(exist_ok=True)
    for name, part in splits.items():
        part[["text", "category", "intent"]].to_json(
            OUT_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
        print(name, len(part), "rows,", part["intent"].nunique(), "intents")


if __name__ == "__main__":
    main()