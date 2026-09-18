"""The one place the pipeline talks to Claude.

A thin wrapper in the sense CLAUDE.md means: the SDK call is here and
nowhere else, what goes in and what comes back are plain data, and the
stages that use it are tested against ``FakeClaude`` with recorded answers.
The wrapper knows the price of a token so a stage can estimate before it
spends and record after; it does not know the ledger -- the stage holds
that, because the stage knows what it is buying.

The key is never an argument. The SDK reads ``ANTHROPIC_API_KEY`` from the
environment, which on the box the bootstrap fills from instance metadata;
a key in config would be a key in git.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Cache pricing relative to the input rate: a write costs a quarter more, a
# read a tenth. Neither is used by a one-shot call, but usage reports them
# and the ledger should not under-count if a caller ever caches.
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.10


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""
    input: float
    output: float

    @classmethod
    def from_cfg(cls, cfg, model: str) -> "Price":
        table = cfg.get("api.prices_usd_per_mtok", {}) or {}
        p = table.get(model)
        if not p:
            raise KeyError(f"api.prices_usd_per_mtok has no price for {model!r}; "
                           f"add one before spending on it")
        return cls(float(p["input"]), float(p["output"]))


def estimate_usd(n_input: int, n_output: int, price: Price) -> float:
    return round((n_input * price.input + n_output * price.output) / 1e6, 4)


def usage_usd(usage: Any, price: Price) -> float:
    """What a response cost, from its ``usage`` block."""
    n_in = int(getattr(usage, "input_tokens", 0) or 0)
    n_out = int(getattr(usage, "output_tokens", 0) or 0)
    n_cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    n_cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    usd = (n_in * price.input + n_out * price.output
           + n_cw * price.input * CACHE_WRITE_FACTOR
           + n_cr * price.input * CACHE_READ_FACTOR) / 1e6
    return round(usd, 4)


@dataclass(frozen=True)
class Completion:
    text: str
    data: Any                 # the parsed JSON
    input_tokens: int
    output_tokens: int
    stop_reason: str
    usd: float
    model: str


class ClaudeRefused(RuntimeError):
    """The model declined (stop_reason refusal). Not retried: the input is
    trek footage and a chat about it, so a refusal is something to read."""


class ClaudeTruncated(RuntimeError):
    """The answer hit max_tokens. Raise beats.max_output_tokens."""


class ClaudeBadJSON(RuntimeError):
    """The text was not JSON despite the output format. The text is kept
    on the exception so it can be written out and read."""
    def __init__(self, text: str, exc: Exception):
        self.text = text
        super().__init__(f"the answer is not JSON ({exc}); first 200 chars: {text[:200]!r}")


class Claude:
    """The SDK behind two methods: how big is this, and answer it as JSON."""

    def __init__(self, model: str, *, price: Price, effort: str = "high",
                 max_tokens: int = 16000):
        self.model = model
        self.price = price
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self._client = None

    @property
    def client(self):
        # Imported here so the package imports, and every stage but this one
        # runs, on a machine without the SDK.
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("the anthropic SDK is not installed: "
                                  "pip install -e '.[api]'") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def count_tokens(self, system: str, messages: Sequence[Mapping[str, Any]]) -> int:
        r = self.client.messages.count_tokens(model=self.model, system=system,
                                              messages=list(messages))
        return int(r.input_tokens)

    def complete_json(self, system: str, messages: Sequence[Mapping[str, Any]],
                      schema: Mapping[str, Any]) -> Completion:
        """One answer, constrained to ``schema`` by the API, streamed so a
        long answer cannot hit the HTTP timeout, parsed before it is returned."""
        with self.client.messages.stream(
                model=self.model, max_tokens=self.max_tokens,
                system=system, messages=list(messages),
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort,
                               "format": {"type": "json_schema", "schema": dict(schema)}},
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            details = getattr(msg, "stop_details", None)
            raise ClaudeRefused(f"{self.model} refused: "
                                f"{getattr(details, 'category', None)} "
                                f"{getattr(details, 'explanation', '') or ''}".strip())
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if msg.stop_reason == "max_tokens":
            raise ClaudeTruncated(f"the answer stopped at {self.max_tokens} tokens; "
                                  f"raise beats.max_output_tokens")
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ClaudeBadJSON(text, exc) from exc
        return Completion(text=text, data=data,
                          input_tokens=int(msg.usage.input_tokens),
                          output_tokens=int(msg.usage.output_tokens),
                          stop_reason=str(msg.stop_reason),
                          usd=usage_usd(msg.usage, self.price), model=self.model)


class FakeClaude:
    """Recorded answers, in order, and a record of what was asked.

    The stage tests run against this; the one live test costs cents and is
    opt-in. ``n_tokens`` is what ``count_tokens`` reports and what each
    completion charges as input; output is a fixed 200 tokens.
    """

    def __init__(self, responses: Sequence[str], *, n_tokens: int = 1000,
                 price: Price = Price(5.0, 25.0), model: str = "fake-claude"):
        self.responses = list(responses)
        self.n_tokens = int(n_tokens)
        self.price = price
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def count_tokens(self, system: str, messages: Sequence[Mapping[str, Any]]) -> int:
        self.calls.append({"op": "count_tokens", "system": system, "messages": list(messages)})
        return self.n_tokens

    def complete_json(self, system: str, messages: Sequence[Mapping[str, Any]],
                      schema: Mapping[str, Any]) -> Completion:
        self.calls.append({"op": "complete_json", "system": system,
                           "messages": list(messages), "schema": dict(schema)})
        if not self.responses:
            raise RuntimeError("FakeClaude has no answers left")
        text = self.responses.pop(0)
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ClaudeBadJSON(text, exc) from exc
        return Completion(text=text, data=data, input_tokens=self.n_tokens,
                          output_tokens=200, stop_reason="end_turn",
                          usd=estimate_usd(self.n_tokens, 200, self.price),
                          model=self.model)
