"""Answer generators. Each adapter returns text plus token counts and latency."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from .. import config


@dataclass(frozen=True)
class Generation:
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str


class NotConfigured(RuntimeError):
    pass


class Generator(Protocol):
    name: str

    def configured(self) -> tuple[bool, str]: ...

    def generate(self, system: str, user: str, max_tokens: int) -> Generation: ...


@lru_cache(maxsize=1)
def prices() -> dict:
    return tomllib.loads(config.PRICES_FILE.read_text())


def cost_usd(model: str, input_tokens: float, output_tokens: float) -> float:
    p = prices()["models"][model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def get(name: str) -> Generator:
    if name == "qwen3-1.7b":
        from .mlx_qwen import QwenMLX

        return QwenMLX()
    if name in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        from .anthropic_api import ClaudeMessages

        return ClaudeMessages(name)
    raise ValueError(f"unknown model {name!r}")
