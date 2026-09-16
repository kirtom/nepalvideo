"""The SQLite layer: schema, migrations, and the generic upsert."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from nepal import db


def test_upsert_refuses_rows_with_different_keys(tmp_path):
    """Columns come from the first row. A row that lacks a key would have
    that column silently dropped for every row -- which is how 930 video
    shots lost their position: the first row had no `lat`."""
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "method": "m"}]          # no confidence
    with pytest.raises(ValueError) as exc:
        db.upsert(conn, "decisions", ["key"], rows)
    assert "confidence" in str(exc.value)


def test_upsert_writes_every_column_of_uniform_rows(tmp_path):
    conn = db.init(tmp_path / "t.sqlite")
    rows = [{"key": "a", "value": "1", "confidence": 1.0, "method": "m"},
            {"key": "b", "value": "2", "confidence": 0.5, "method": "n"}]
    assert db.upsert(conn, "decisions", ["key"], rows) == 2
    got = {r["key"]: r["confidence"] for r in conn.execute("SELECT key, confidence FROM decisions")}
    assert got == {"a": 1.0, "b": 0.5}
