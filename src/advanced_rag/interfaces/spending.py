"""Session-local spending guards for the Streamlit interface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpendingLimits:
    """User-selected session limits; zero disables an individual limit."""

    questions: int
    api_calls: int
    tokens: int


@dataclass(frozen=True)
class SessionUsage:
    """Cumulative chat-provider-reported usage for one browser session."""

    questions: int = 0
    api_calls: int = 0
    tokens: int = 0

    def add(self, *, api_calls: int, tokens: int) -> "SessionUsage":
        return SessionUsage(
            questions=self.questions + 1,
            api_calls=self.api_calls + api_calls,
            tokens=self.tokens + tokens,
        )


@dataclass(frozen=True)
class SpendingDecision:
    """Whether another question may start and why it may be blocked."""

    allowed: bool
    reasons: tuple[str, ...]
    remaining_questions: int | None
    remaining_api_calls: int | None
    remaining_tokens: int | None


def spending_decision(
    limits: SpendingLimits,
    usage: SessionUsage,
    *,
    api_call_reserve: int,
    token_reserve: int,
) -> SpendingDecision:
    """Reserve a conservative worst-case allowance before starting a graph run."""
    remaining_questions = _remaining(limits.questions, usage.questions)
    remaining_api_calls = _remaining(limits.api_calls, usage.api_calls)
    remaining_tokens = _remaining(limits.tokens, usage.tokens)
    reasons: list[str] = []
    if remaining_questions is not None and remaining_questions < 1:
        reasons.append("The session question limit has been reached.")
    if remaining_api_calls is not None and remaining_api_calls < api_call_reserve:
        reasons.append(
            f"Fewer than {api_call_reserve} chat API calls remain, which is below "
            "the graph reserve."
        )
    if remaining_tokens is not None and remaining_tokens < token_reserve:
        reasons.append(
            f"Fewer than {token_reserve:,} tokens remain, which is below the conservative "
            "per-question reserve."
        )
    return SpendingDecision(
        allowed=not reasons,
        reasons=tuple(reasons),
        remaining_questions=remaining_questions,
        remaining_api_calls=remaining_api_calls,
        remaining_tokens=remaining_tokens,
    )


def _remaining(limit: int, used: int) -> int | None:
    return None if limit == 0 else max(0, limit - used)
