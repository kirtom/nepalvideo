"""The Claude wrapper: prices, the fake, and the SDK call shape -- no network."""
import json
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from nepal.cloud import claude as cl
from nepal.config import Config

PRICE = cl.Price(5.0, 25.0)


def test_prices_come_from_the_config_and_an_unpriced_model_is_refused():
    cfg = Config({"api": {"prices_usd_per_mtok": {"claude-opus-5": {"input": 5, "output": 25}}}})
    assert cl.Price.from_cfg(cfg, "claude-opus-5") == PRICE
    with pytest.raises(KeyError):
        cl.Price.from_cfg(cfg, "claude-mystery-9")


def test_estimate_and_usage_arithmetic():
    assert cl.estimate_usd(100_000, 10_000, PRICE) == 0.75
    usage = SimpleNamespace(input_tokens=100_000, output_tokens=10_000,
                            cache_creation_input_tokens=0, cache_read_input_tokens=0)
    assert cl.usage_usd(usage, PRICE) == 0.75
    cached = SimpleNamespace(input_tokens=0, output_tokens=0,
                             cache_creation_input_tokens=1_000_000,
                             cache_read_input_tokens=1_000_000)
    assert cl.usage_usd(cached, PRICE) == 5.0 * 1.25 + 5.0 * 0.10


def test_the_fake_answers_in_order_and_records_what_it_was_asked():
    fake = cl.FakeClaude(['{"a": 1}', '{"a": 2}'], n_tokens=50_000)
    assert fake.count_tokens("sys", [{"role": "user", "content": "hi"}]) == 50_000
    c1 = fake.complete_json("sys", [{"role": "user", "content": "hi"}], {"type": "object"})
    c2 = fake.complete_json("sys", [], {"type": "object"})
    assert (c1.data, c2.data) == ({"a": 1}, {"a": 2})
    assert c1.usd == cl.estimate_usd(50_000, 200, PRICE)
    assert [c["op"] for c in fake.calls] == ["count_tokens", "complete_json", "complete_json"]
    with pytest.raises(RuntimeError):
        fake.complete_json("sys", [], {})
    with pytest.raises(cl.ClaudeBadJSON):
        cl.FakeClaude(["not json"]).complete_json("s", [], {})


def test_the_module_imports_without_the_sdk_and_says_so_when_used(monkeypatch):
    c = cl.Claude("claude-opus-5", price=PRICE)          # no SDK touched yet
    monkeypatch.setitem(sys.modules, "anthropic", None)  # "not installed"
    with pytest.raises(ImportError) as exc:
        _ = c.client
    assert "pip install -e '.[api]'" in str(exc.value)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def get_final_message(self):
        return self.msg


def _client(msg, seen):
    @contextmanager
    def stream(**kw):
        seen.append(kw)
        yield _Stream(msg)

    def count_tokens(**kw):
        seen.append(kw)
        return SimpleNamespace(input_tokens=1234)
    return SimpleNamespace(messages=SimpleNamespace(stream=stream, count_tokens=count_tokens))


def _msg(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop, stop_details=None,
        content=[SimpleNamespace(type="thinking", thinking=""),
                 SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=100,
                              cache_creation_input_tokens=0, cache_read_input_tokens=0))


def test_complete_json_streams_with_the_schema_and_the_effort_and_parses():
    seen = []
    c = cl.Claude("claude-opus-5", price=PRICE, effort="xhigh", max_tokens=9000)
    c._client = _client(_msg('{"title": "x"}'), seen)
    out = c.complete_json("SYS", [{"role": "user", "content": "go"}],
                          {"type": "object", "properties": {"title": {"type": "string"}}})
    assert out.data == {"title": "x"} and out.usd == cl.estimate_usd(1000, 100, PRICE)
    kw = seen[-1]
    assert kw["model"] == "claude-opus-5" and kw["max_tokens"] == 9000
    assert kw["system"] == "SYS" and kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "xhigh"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["format"]["schema"]["properties"]["title"] == {"type": "string"}
    assert c.count_tokens("SYS", []) == 1234


def test_refusal_truncation_and_bad_json_are_distinct_errors():
    c = cl.Claude("claude-opus-5", price=PRICE)
    c._client = _client(_msg("", stop="refusal"), [])
    with pytest.raises(cl.ClaudeRefused):
        c.complete_json("s", [], {})
    c._client = _client(_msg('{"partial', stop="max_tokens"), [])
    with pytest.raises(cl.ClaudeTruncated):
        c.complete_json("s", [], {})
    c._client = _client(_msg("nope"), [])
    with pytest.raises(cl.ClaudeBadJSON) as exc:
        c.complete_json("s", [], {})
    assert exc.value.text == "nope"
