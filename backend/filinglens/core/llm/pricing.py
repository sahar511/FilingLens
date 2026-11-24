"""USD per million tokens. Check the provider's official pricing page and update these;
prices change often. Unknown models are logged with cost=None rather than guessed."""

from __future__ import annotations

from .base import Usage

# model: (input, output, cached_input)
PRICES: dict[str, tuple[float, float, float]] = {
    # Rate for prompts up to 100k tokens; cached rate assumed at 10% of input.
    # Verify both on the official pricing page.
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "llama-3.3-70b-versatile": (0.59, 0.79, 0.59),
    # Add the models you use, e.g.:
    # "claude-sonnet-5-5": (input, output, cached_input),
}


def cost_usd(model: str, usage: Usage) -> float | None:
    key = next((k for k in PRICES if model.startswith(k)), None)
    if key is None:
        return None
    p_in, p_out, p_cache = PRICES[key]
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached * p_in + usage.cached_input_tokens * p_cache + usage.output_tokens * p_out
    ) / 1_000_000
