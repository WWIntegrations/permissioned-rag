"""Generation through the Anthropic Messages API (Claude Haiku 5.5, Claude Sonnet 5.5).

The API key is read from the ANTHROPIC_API_KEY environment variable and from
nowhere else; if it is absent the adapter reports "not configured" and makes
no request.

Request shape: effort "low" and thinking switched off, because the task is
short extractive answering and thinking tokens would be billed as output.
Claude Haiku 5.5 accepts thinking {"type": "disabled"} at effort high or
below; Claude Sonnet 5.5 rejects "disabled" and uses {"type": "between_tools"}
for the same purpose. Sampling parameters (temperature, top_p) are not sent:
current models reject them.
"""

from __future__ import annotations

import os
import time

from . import Generation, NotConfigured, prices

THINKING_OFF = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}


class ClaudeMessages:
    def __init__(self, name: str, client=None):
        if name not in THINKING_OFF:
            raise ValueError(name)
        self.name = name
        self.api_id = prices()["models"][name]["api_id"]
        self._client = client

    def configured(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, "injected client"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "not configured: anthropic package missing (uv sync --extra anthropic)"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "not configured: ANTHROPIC_API_KEY is not set"
        return True, "ANTHROPIC_API_KEY"

    def client(self):
        if self._client is None:
            ok, why = self.configured()
            if not ok:
                raise NotConfigured(why)
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=3)
        return self._client

    def request(self, system: str, user: str, max_tokens: int) -> dict:
        return {
            "model": self.api_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "thinking": THINKING_OFF[self.name],
            "output_config": {"effort": "low"},
        }

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        client = self.client()
        t0 = time.perf_counter()
        resp = client.messages.create(**self.request(system, user, max_tokens))
        latency = time.perf_counter() - t0
        if resp.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return Generation(
            model=self.name,
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
            stop_reason=resp.stop_reason or "",
        )
