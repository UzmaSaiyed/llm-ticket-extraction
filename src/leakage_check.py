import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.metrics import parse_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    train = pd.read_json("data/train.jsonl", lines=True)
    preds = pd.read_json(args.predictions, lines=True)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    train_matrix = vectorizer.fit_transform(train["text"])
    eval_matrix = vectorizer.transform(preds["text"])
    preds["similarity"] = cosine_similarity(eval_matrix, train_matrix).max(axis=1)

    preds["correct"] = [
        (parse_prediction(p) or {}).get("intent") == gold
        for p, gold in zip(preds["prediction"], preds["intent"])
    ]
    preds["group"] = pd.cut(
        preds["similarity"], [0, 0.6, 0.8, 0.9, 1.0], include_lowest=True
    )
    print(preds.groupby("group", observed=True)["correct"].agg(["count", "mean"]))
    print("median similarity to the closest training message:", preds["similarity"].median())


if __name__ == "__main__":
    main()