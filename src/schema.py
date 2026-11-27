"""Allowed values for the extraction output."""

INTENT_TO_CATEGORY = {
    "create_account": "ACCOUNT",
    "delete_account": "ACCOUNT",
    "edit_account": "ACCOUNT",
    "switch_account": "ACCOUNT",
    "registration_problems": "ACCOUNT",
    "recover_password": "ACCOUNT",
    "cancel_order": "ORDER",
    "change_order": "ORDER",
    "place_order": "ORDER",
    "track_order": "ORDER",
    "check_refund_policy": "REFUND",
    "get_refund": "REFUND",
    "track_refund": "REFUND",
    "contact_customer_service": "CONTACT",
    "contact_human_agent": "CONTACT",
    "check_invoice": "INVOICE",
    "get_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT",
    "payment_issue": "PAYMENT",
    "complaint": "FEEDBACK",
    "review": "FEEDBACK",
    "delivery_options": "DELIVERY",
    "delivery_period": "DELIVERY",
    "change_shipping_address": "SHIPPING",
    "set_up_shipping_address": "SHIPPING",
    "newsletter_subscription": "SUBSCRIPTION",
    "check_cancellation_fee": "CANCEL",
}

INTENTS = sorted(INTENT_TO_CATEGORY)
CATEGORIES = sorted(set(INTENT_TO_CATEGORY.values()))