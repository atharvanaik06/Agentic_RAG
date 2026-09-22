from advanced_rag.interfaces.spending import (
    SessionUsage,
    SpendingLimits,
    spending_decision,
)


def test_spending_guard_reserves_calls_and_tokens_before_question() -> None:
    limits = SpendingLimits(questions=3, api_calls=4, tokens=10000)
    usage = SessionUsage(questions=1, api_calls=3, tokens=5000)

    decision = spending_decision(
        limits,
        usage,
        api_call_reserve=2,
        token_reserve=4000,
    )

    assert decision.allowed is False
    assert decision.remaining_questions == 2
    assert decision.remaining_api_calls == 1
    assert decision.remaining_tokens == 5000
    assert "chat API calls" in decision.reasons[0]


def test_spending_guard_stops_at_question_and_token_limits() -> None:
    decision = spending_decision(
        SpendingLimits(questions=1, api_calls=10, tokens=3000),
        SessionUsage(questions=1, api_calls=1, tokens=2500),
        api_call_reserve=2,
        token_reserve=1000,
    )

    assert decision.allowed is False
    assert len(decision.reasons) == 2
    assert decision.remaining_questions == 0
    assert decision.remaining_tokens == 500


def test_zero_limits_are_unlimited_and_usage_accumulates() -> None:
    usage = SessionUsage().add(api_calls=2, tokens=1234)
    decision = spending_decision(
        SpendingLimits(questions=0, api_calls=0, tokens=0),
        usage,
        api_call_reserve=5,
        token_reserve=99999,
    )

    assert usage == SessionUsage(questions=1, api_calls=2, tokens=1234)
    assert decision.allowed is True
    assert decision.remaining_questions is None
    assert decision.remaining_api_calls is None
    assert decision.remaining_tokens is None


def test_multi_question_run_reserves_and_records_every_case() -> None:
    decision = spending_decision(
        SpendingLimits(questions=5, api_calls=20, tokens=50000),
        SessionUsage(questions=3, api_calls=1, tokens=100),
        question_reserve=3,
        api_call_reserve=6,
        token_reserve=10000,
    )
    usage = SessionUsage().add(questions=3, api_calls=5, tokens=9000)

    assert decision.allowed is False
    assert "3 questions" in decision.reasons[0]
    assert usage == SessionUsage(questions=3, api_calls=5, tokens=9000)
