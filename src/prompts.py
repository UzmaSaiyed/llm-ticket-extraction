import json

from src.schema import INTENT_TO_CATEGORY

EXAMPLES = [
    ("how do i get my money back for order 5521", "REFUND", "get_refund"),
    ("where is the refund you promised me last week", "REFUND", "track_refund"),
    ("can i talk to a real person please", "CONTACT", "contact_human_agent"),
    ("send me a copy of my last bill", "INVOICE", "get_invoice"),
    ("i want to change the address my parcel goes to", "SHIPPING", "change_shipping_address"),
]

INTENT_HINTS = {
    "get_invoice": "the customer wants to receive a copy of an invoice or bill",
    "check_invoice": "the customer wants to look at or find an invoice or bill",
    "get_refund": "the customer wants to ask for a refund",
    "track_refund": "the customer asks about the status of a refund already requested",
    "check_refund_policy": "the customer asks about the rules, conditions or timing of refunds",
    "change_shipping_address": "the customer wants to correct or change an address already entered",
    "set_up_shipping_address": "the customer wants to enter or add a new address",
    "contact_human_agent": "the customer wants to speak with a person",
    "contact_customer_service": "the customer wants to reach customer service or asks for its contact details or opening hours",
}

def _allowed_values() -> str:
    by_category: dict[str, list[str]] = {}
    for intent, category in INTENT_TO_CATEGORY.items():
        by_category.setdefault(category, []).append(intent)
    return "\n".join(
        f"{category}: {', '.join(sorted(intents))}"
        for category, intents in sorted(by_category.items())
    )


def _hints() -> str:
    return "\n".join(f"- {intent}: {hint}" for intent, hint in INTENT_HINTS.items())


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
            "Some intents look alike. Tell them apart like this:\n"
            f"{_hints()}\n\n"
            "Rules:\n"
            "- Reply with one JSON object only, in the form "
            '{"category": "...", "intent": "..."}.\n'
            "- The intent must be one of the listed names. Never make up a new name.\n"
            "- Copy the spelling from the list exactly. Add no other text.\n\n"
            f"Examples:\n{examples}\n\n"
            f"Message: {text}"
        )
    raise ValueError(f"Unknown prompt kind: {kind}")