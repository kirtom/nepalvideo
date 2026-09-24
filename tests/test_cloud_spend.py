"""The spend ledger: what the project has paid for, and the ceiling."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
import shutil

import pytest

from nepal.cloud import spend


def test_a_fresh_ledger_is_empty(tmp_path):
    led = spend.Ledger(tmp_path / "spend")
    assert led.entries == [] and led.total() == 0.0


def test_record_persists_one_file_per_entry_and_totals(tmp_path):
    led = spend.Ledger(tmp_path / "spend")
    led.record("gce:cpu", 0.42, detail="3.8 h e2-standard-8 spot")
    led.record("api:captions", 1.31)
    files = sorted((tmp_path / "spend").glob("*.json"))
    assert len(files) == 2
    assert {f.name.split("_")[1] for f in files} == {"gce-cpu", "api-captions"}
    again = spend.Ledger(tmp_path / "spend")
    assert [e.what for e in again.entries] == ["gce:cpu", "api:captions"]   # recorded order
    assert again.total() == pytest.approx(1.73)
    raw = json.loads(next(f for f in files if "gce-cpu" in f.name).read_text())
    assert raw["usd"] == 0.42 and raw["when"]


def test_two_hosts_entries_merge_by_union_after_any_rsync(tmp_path):
    """The box and this machine each write their own files; copying the
    directories over each other in either order loses nothing."""
    here = spend.Ledger(tmp_path / "here" / "spend")
    here.record("gce:cpu", 0.15)
    box = spend.Ledger(tmp_path / "box" / "spend")
    box.record("beats", 0.9)
    # "rsync" both ways, no deletes
    for src, dst in ((tmp_path / "here" / "spend", tmp_path / "box" / "spend"),
                     (tmp_path / "box" / "spend", tmp_path / "here" / "spend")):
        for f in src.glob("*.json"):
            shutil.copy(f, dst / f.name)
    assert spend.Ledger(tmp_path / "here" / "spend").total() == pytest.approx(1.05)
    assert spend.Ledger(tmp_path / "box" / "spend").total() == pytest.approx(1.05)


def test_the_legacy_single_file_is_read_and_never_written(tmp_path):
    (tmp_path / "spend.json").write_text(json.dumps(
        {"entries": [{"when": "2026-09-18T06:21:33+00:00", "what": "gce:cpu",
                      "usd": 0.1322, "detail": "old"}], "total_usd": 0.1322}))
    led = spend.Ledger(tmp_path / "spend")
    assert led.total() == pytest.approx(0.1322)
    led.record("beats", 1.0)
    assert spend.Ledger(tmp_path / "spend").total() == pytest.approx(1.1322)
    assert json.loads((tmp_path / "spend.json").read_text())["total_usd"] == 0.1322


def test_guard_refuses_when_the_estimate_would_cross_the_ceiling(tmp_path):
    led = spend.Ledger(tmp_path / "spend")
    led.record("api:beats", 13.0)
    led.guard(1.5, ceiling_usd=15.0)                  # 14.5: fine
    with pytest.raises(spend.SpendCeiling) as exc:
        led.guard(2.5, ceiling_usd=15.0)              # 15.5: not fine
    assert exc.value.total == 13.0 and exc.value.ceiling == 15.0


def test_guard_counts_the_hours_nothing_has_booked_yet(tmp_path):
    """VM hours are recorded at `remote down`. A box that is up has spent
    money no entry holds, and that is what crossed the ceiling in 2026-09:
    the ledger read 5.21 for four days while the box billed 104 hours."""
    led = spend.Ledger(tmp_path / "spend")
    led.record("api:beats", 13.0)
    led.guard(1.0, ceiling_usd=15.0, unbooked_usd=0.5)          # 14.5: fine
    with pytest.raises(spend.SpendCeiling) as exc:
        led.guard(1.0, ceiling_usd=15.0, unbooked_usd=1.5)      # 15.5: not fine
    assert exc.value.total == pytest.approx(14.5)               # the live total, not 13.0
    assert "14.50 USD spent" in str(exc.value)


def test_a_corrupt_entry_is_an_error_not_a_zero(tmp_path):
    (tmp_path / "spend").mkdir()
    (tmp_path / "spend" / "x.json").write_text("{not json")
    with pytest.raises(ValueError):
        spend.Ledger(tmp_path / "spend")
