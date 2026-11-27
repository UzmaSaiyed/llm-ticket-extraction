from src.metrics import parse_prediction, score

GOLD = {"category": "ORDER", "intent": "cancel_order"}


def test_parse_finds_json_inside_extra_text():
    text = 'Sure! {"category": "ORDER", "intent": "cancel_order"} hope that helps'
    assert parse_prediction(text) == GOLD


def test_parse_returns_none_for_invalid_json():
    assert parse_prediction("category: ORDER") is None
    assert parse_prediction('{"category": "ORDER", ') is None


def test_perfect_prediction():
    result = score(['{"category": "ORDER", "intent": "cancel_order"}'], [GOLD])
    assert result["exact_match"] == 1.0
    assert result["field_f1"] == 1.0
    assert result["json_validity"] == 1.0


def test_wrong_intent_keeps_category_credit():
    result = score(['{"category": "ORDER", "intent": "track_order"}'], [GOLD])
    assert result["exact_match"] == 0.0
    assert result["category_accuracy"] == 1.0
    assert result["intent_accuracy"] == 0.0


def test_invalid_json_scores_zero():
    result = score(["I think it is a cancellation"], [GOLD])
    assert result["json_validity"] == 0.0
    assert result["exact_match"] == 0.0


def test_missing_field_lowers_recall_not_precision():
    result = score(['{"intent": "cancel_order"}'], [GOLD])
    assert result["field_precision"] == 1.0
    assert result["field_recall"] == 0.5