"""Hosted adapter wiring and price config. No network calls are made."""

from __future__ import annotations

import pytest

from prag.generation import cost_usd, prices
from prag.generation.anthropic_api import ClaudeMessages


def test_not_configured_without_env_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, why = ClaudeMessages("claude-haiku-5-5").configured()
    assert not ok and "not configured" in why


@pytest.mark.parametrize(
    "model,thinking", [("claude-haiku-5-5", "disabled"), ("claude-sonnet-5-5", "between_tools")]
)
def test_request_shape(model, thinking):
    req = ClaudeMessages(model).request("sys", "user", 300)
    assert req["model"] == model
    assert req["thinking"] == {"type": thinking}
    assert req["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k"} & set(req)


def test_generate_uses_injected_client_and_reports_usage():
    class Usage:
        input_tokens, output_tokens = 1000, 50

    class Block:
        type, text = "text", "Answer [#1]."

    class Resp:
        stop_reason, content, usage = "end_turn", [Block()], Usage()

    class Messages:
        def create(self, **kw):
            self.kw = kw
            return Resp()

    class Client:
        messages = Messages()

    g = ClaudeMessages("claude-sonnet-5-5", client=Client()).generate("s", "u", 100)
    assert g.text == "Answer [#1]." and g.input_tokens == 1000 and g.output_tokens == 50


def test_prices_are_dated_and_cost_math():
    p = prices()
    assert p["checked"]
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 0) == pytest.approx(p["models"]["claude-sonnet-5-5"]["input"])
    assert cost_usd("claude-haiku-5-5", 54_643, 5_528) == pytest.approx(0.0082, abs=1e-4)
    assert cost_usd("qwen3-1.7b", 10**9, 10**9) == 0.0
