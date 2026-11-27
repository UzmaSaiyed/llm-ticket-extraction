import json
import re

FIELDS = ("category", "intent")


def parse_prediction(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def score(predictions: list[str], golds: list[dict]) -> dict[str, float]:
    total = len(golds)
    valid = exact = matched = predicted = 0
    per_field = {field: 0 for field in FIELDS}

    for raw, gold in zip(predictions, golds):
        pred = parse_prediction(raw)
        if pred is None:
            continue
        valid += 1
        correct = [f for f in FIELDS if f in pred and pred[f] == gold[f]]
        predicted += sum(1 for f in FIELDS if f in pred)
        matched += len(correct)
        exact += len(correct) == len(FIELDS)
        for field in correct:
            per_field[field] += 1

    precision = matched / predicted if predicted else 0.0
    recall = matched / (total * len(FIELDS))
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0
    return {
        "json_validity": valid / total,
        "exact_match": exact / total,
        "category_accuracy": per_field["category"] / total,
        "intent_accuracy": per_field["intent"] / total,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": f1,
    }