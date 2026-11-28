import pytest

from src.prompts import EXAMPLES, INTENT_HINTS, build_prompt
from src.schema import INTENT_TO_CATEGORY, INTENTS


def test_engineered_prompt_lists_every_intent():
    prompt = build_prompt("engineered", "hello")
    assert all(intent in prompt for intent in INTENTS)


def test_examples_use_valid_labels():
    for _, category, intent in EXAMPLES:
        assert INTENT_TO_CATEGORY[intent] == category


def test_message_is_included_in_both_prompts():
    assert "cancel my order" in build_prompt("simple", "cancel my order")
    assert "cancel my order" in build_prompt("engineered", "cancel my order")


def test_unknown_prompt_kind_is_rejected():
    with pytest.raises(ValueError):
        build_prompt("other", "hello")


def test_hints_use_valid_intents():
    assert set(INTENT_HINTS) <= set(INTENTS)