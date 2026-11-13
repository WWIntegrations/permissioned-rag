"""Local generation with Qwen3-1.7B (4-bit) through MLX on Apple silicon.

Greedy decoding (the mlx-lm default sampler is temperature 0) and Qwen3's
thinking mode switched off through the chat template, so outputs are
repeatable and the token counts are answer tokens only.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache

from .. import config
from . import Generation

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@lru_cache(maxsize=1)
def _load():
    from mlx_lm import load

    return load(config.QWEN_MODEL)


class QwenMLX:
    name = "qwen3-1.7b"

    def configured(self) -> tuple[bool, str]:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            return False, "mlx-lm is not installed (uv sync --extra mlx)"
        return True, "local"

    def prompt(self, system: str, user: str) -> str:
        _, tok = _load()
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )

    def count_tokens(self, text: str) -> int:
        _, tok = _load()
        return len(tok.encode(text))

    def generate(self, system: str, user: str, max_tokens: int) -> Generation:
        from mlx_lm import stream_generate

        model, tok = _load()
        prompt = self.prompt(system, user)
        t0 = time.perf_counter()
        parts: list[str] = []
        last = None
        for last in stream_generate(model, tok, prompt, max_tokens=max_tokens):
            parts.append(last.text)
        latency = time.perf_counter() - t0
        text = "".join(parts).strip()
        return Generation(
            model=self.name,
            text=text,
            input_tokens=last.prompt_tokens if last else 0,
            output_tokens=last.generation_tokens if last else 0,
            latency_s=latency,
            stop_reason=(last.finish_reason or "") if last else "",
        )
