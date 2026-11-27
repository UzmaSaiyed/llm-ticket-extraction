import json

from src.schema import INTENT_TO_CATEGORY

EXAMPLES = [
    ("how do i get my money back for order 5521", "REFUND", "get_refund"),
    ("where is the refund you promised me last week", "REFUND", "track_refund"),
    ("can i talk to a real person please", "CONTACT", "contact_human_agent"),
    ("send me a copy of my last bill", "INVOICE", "get_invoice"),
    ("i want to change the address my parcel goes to", "SHIPPING", "change_shipping_address"),
]


def _allowed_values() -> str:
    by_category: dict[str, list[str]] = {}
    for intent, category in INTENT_TO_CATEGORY.items():
        by_category.setdefault(category, []).append(intent)
    return "\n".join(
        f"{category}: {', '.join(sorted(intents))}"
        for category, intents in sorted(by_category.items())
    )


def build_prompt(kind: str, text: str) -> str:
    if kind == "simple":
        return (
            "Classify this customer message. "
            "Reply with JSON containing category and intent.\n"
            f"Message: {text}"
        )
    if kind == "engineered":
        examples = "\n".join(
            f"Message: {example}\n{json.dumps({'category': category, 'intent': intent})}"
            for example, category, intent in EXAMPLES
        )
        return (
            "You label customer-support messages.\n\n"
            "Choose exactly one intent from the list below. The category is the "
            "heading that the intent is listed under.\n"
            f"{_allowed_values()}\n\n"
            "Rules:\n"
            "- Reply with one JSON object only, in the form "
            '{"category": "...", "intent": "..."}.\n'
            "- Copy the spelling from the list exactly. Add no other text.\n\n"
            f"Examples:\n{examples}\n\n"
            f"Message: {text}"
        )
    raise ValueError(f"Unknown prompt kind: {kind}")