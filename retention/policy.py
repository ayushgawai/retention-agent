"""Deterministic retention policy. The executor calls `check` before every tool runs."""
from dataclasses import dataclass

PLANS = {"basic": 8.99, "standard": 15.49, "premium": 22.99}
MIN_TENURE_FOR_DISCOUNT = 3   # months
MAX_DISCOUNT_PCT = 30
MAX_OFFERS = 1                # retention offers per conversation
REFUND_WINDOW_DAYS = 14
MAX_PAUSE_MONTHS = 2
MAX_VERIFY_ATTEMPTS = 2

OFFERS = {"offer_discount", "pause_subscription", "change_plan"}
CHANGES = OFFERS | {"issue_refund", "cancel_subscription"}


@dataclass
class State:
    """What the policy needs to know about the conversation so far."""
    verified: bool = False
    verify_failures: int = 0
    offers: int = 0
    repeat_cancel: bool = False   # customer said cancel again after an offer
    refunded: float = 0.0


def check(account, state: State, name: str, args: dict) -> str:
    """Return the reason a call is refused, or an empty string if it may run."""
    if state.repeat_cancel and name not in ("cancel_subscription", "lookup_account", "verify_identity"):
        return "customer_repeated_cancel"  # no further pitching, no stalling
    if name == "verify_identity" and state.verify_failures >= MAX_VERIFY_ATTEMPTS:
        return "verification_locked"
    if name in CHANGES and not state.verified:
        return "identity_not_verified"
    if name in OFFERS and state.offers >= MAX_OFFERS:
        return "one_offer_limit"
    if name == "offer_discount":
        if account.tenure_months < MIN_TENURE_FOR_DISCOUNT:
            return "tenure_below_3_months"
        if not 0 < args["percent"] <= MAX_DISCOUNT_PCT:
            return "discount_over_cap"
    if name == "pause_subscription" and not 1 <= args["months"] <= MAX_PAUSE_MONTHS:
        return "pause_over_2_months"
    if name == "change_plan" and (args["plan"] not in PLANS or args["plan"] == account.plan):
        return "invalid_plan"
    if name == "issue_refund":
        if account.days_since_charge > REFUND_WINDOW_DAYS:
            return "outside_refund_window"
        if not 0 < args["amount"] <= account.last_charge - state.refunded:
            return "refund_over_last_charge"
    return ""
