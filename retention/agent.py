"""Plan, check, execute loop with a step budget."""
from dataclasses import dataclass, field

from .simulator import Customer, Scenario
from .tools import Executor


@dataclass
class Outcome:
    sid: str
    status: str
    steps: int
    blocked: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    env: Customer = None
    audit: list = field(default_factory=list)


def run_conversation(sc: Scenario, planner, trial: int = 0, enforce: bool = True, max_steps: int = 8) -> Outcome:
    env = Customer(sc, trial)
    ex = Executor(env, enforce)
    transcript = [("customer", env.first_message)]
    blocked = []

    def finish(step, reason):
        # A customer who repeated "cancel" gets cancelled, not handed off to someone who might pitch again.
        if env.state.repeat_cancel and env.state.verified:
            ex.run("cancel_subscription", {})
        else:
            ex.run("escalate", {"reason": reason})
        return Outcome(sc.sid, env.status, step, blocked, transcript, env, ex.audit)

    for step in range(1, max_steps + 1):
        call = planner.plan(transcript)
        res = ex.run(call.name, call.args)
        transcript.append(("tool", call, res.output))
        if res.customer:
            transcript.append(("customer", res.customer))
        if res.blocked:
            blocked.append(res.blocked)
            if len(blocked) >= 2:  # repeated refusals: stop instead of looping
                return finish(step, "repeated policy refusals")
            continue
        if res.terminal:
            return Outcome(sc.sid, env.status, step, blocked, transcript, env, ex.audit)
    return finish(max_steps, "step budget exhausted")
