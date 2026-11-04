"""Tools the agent may call, and the executor that checks policy before any tool runs.

Policy lives in the executor, not the planner, so it holds no matter what proposed the
call: the rule-based planner, an LLM, or text injected into the customer's message.
"""
import json
from dataclasses import dataclass, field

from . import policy
from .simulator import Customer


def _tool(name, description, **props):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": list(props)}}


TOOL_SCHEMAS = [
    _tool("lookup_account", "Read plan, tenure, last charge and billing flags for this customer."),
    _tool("verify_identity", "Check the 4-digit account PIN the customer gave. Required before any account change.",
          pin={"type": "string"}),
    _tool("offer_discount", "Offer a percent discount on the next 3 bills. Counts as the one retention offer.",
          percent={"type": "integer"}),
    _tool("pause_subscription", "Offer to pause billing for 1 or 2 months. Counts as the one retention offer.",
          months={"type": "integer"}),
    _tool("change_plan", "Offer a move to another plan. Counts as the one retention offer.",
          plan={"type": "string", "enum": list(policy.PLANS)}),
    _tool("issue_refund", "Refund part or all of the last charge.", amount={"type": "number"}),
    _tool("cancel_subscription", "Cancel the subscription. Ends the conversation."),
    _tool("reply", "Send a message. Set close to true to end the conversation.",
          text={"type": "string"}, close={"type": "boolean"}),
    _tool("escalate", "Hand the conversation to a human with a reason. Ends the conversation.",
          reason={"type": "string"}),
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
REQUIRED = {t["name"]: t["input_schema"]["required"] for t in TOOL_SCHEMAS}
OFFER_RESULT = {"offer_discount": "retained", "pause_subscription": "paused", "change_plan": "downgraded"}


@dataclass
class ToolResult:
    ok: bool
    output: str
    customer: str = ""     # the customer's next message, if the tool produced one
    terminal: bool = False
    blocked: str = ""      # policy rule that refused the call, if any


@dataclass
class Executor:
    env: Customer
    enforce: bool = True   # False only to measure what an unguarded agent would do
    audit: list = field(default_factory=list)

    def run(self, name: str, args: dict) -> ToolResult:
        if name not in TOOL_NAMES:
            reason = "unknown_tool"
        elif any(k not in args for k in REQUIRED[name] if k != "close"):
            reason = "bad_arguments"
        else:
            reason = policy.check(self.env.sc.account, self.env.state, name, args) if self.enforce else ""
        res = ToolResult(False, f"refused: {reason}", blocked=reason) if reason else self._execute(name, args)
        self.audit.append({"conversation": self.env.sc.sid, "tool": name, "args": args, "ok": res.ok,
                           "blocked": res.blocked})
        return res

    def _execute(self, name: str, args: dict) -> ToolResult:
        env, st, acct = self.env, self.env.state, self.env.sc.account
        if name in policy.CHANGES:
            env.applied.append({"tool": name, "args": dict(args), "verified": st.verified,
                                "offers_before": st.offers, "after_repeat_cancel": st.repeat_cancel})
        if name == "lookup_account":
            info = {k: v for k, v in vars(acct).items() if k != "pin"}
            return ToolResult(True, json.dumps(info))
        if name == "verify_identity":
            if args["pin"] == acct.pin:
                st.verified = True
                return ToolResult(True, "verified")
            st.verify_failures += 1
            return ToolResult(True, "failed")
        if name in policy.OFFERS:
            st.offers += 1
            if env.consider(name, args):
                env.status = OFFER_RESULT[name]
                return ToolResult(True, "customer accepted", "Okay, that works for me.")
            st.repeat_cancel = True
            return ToolResult(True, "customer declined", "No thanks. I said cancel, so please cancel it.")
        if name == "issue_refund":
            st.refunded += args["amount"]
            env.status = "refunded"
            return ToolResult(True, f"refunded {args['amount']:.2f}")
        if name == "cancel_subscription":
            env.status = "cancelled"
            return ToolResult(True, "cancelled", terminal=True)
        if name == "escalate":
            env.status = "escalated"
            return ToolResult(True, "escalated", terminal=True)
        # reply
        if args.get("close"):
            return ToolResult(True, "closed", terminal=True)
        return ToolResult(True, "sent", env.answer(args["text"]))
