"""The spend ledger: what the project has paid for, and the ceiling."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import pytest

from nepal.cloud import spend


def test_a_fresh_ledger_is_empty(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    assert led.entries == [] and led.total() == 0.0


def test_record_persists_and_totals(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    led.record("gce:cpu", 0.42, detail="3.8 h e2-standard-8 spot")
    led.record("api:captions", 1.31)
    again = spend.Ledger(tmp_path / "spend.json")
    assert [e.what for e in again.entries] == ["gce:cpu", "api:captions"]
    assert again.total() == pytest.approx(1.73)
    raw = json.loads((tmp_path / "spend.json").read_text())
    assert raw["entries"][0]["usd"] == 0.42 and raw["entries"][0]["when"]


def test_guard_refuses_when_the_estimate_would_cross_the_ceiling(tmp_path):
    led = spend.Ledger(tmp_path / "spend.json")
    led.record("api:beats", 13.0)
    led.guard(1.5, ceiling_usd=15.0)                  # 14.5: fine
    with pytest.raises(spend.SpendCeiling) as exc:
        led.guard(2.5, ceiling_usd=15.0)              # 15.5: not fine
    assert exc.value.total == 13.0 and exc.value.ceiling == 15.0


def test_a_corrupt_ledger_is_an_error_not_a_zero(tmp_path):
    (tmp_path / "spend.json").write_text("{not json")
    with pytest.raises(ValueError):
        spend.Ledger(tmp_path / "spend.json")
