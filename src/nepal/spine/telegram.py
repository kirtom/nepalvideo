"""S02.5 -- parse the Telegram export.

The highest-value input per byte in the project: timeline anchors, the
vocabulary that makes S04's captions come back in the operator's own language,
and the on-screen text for Act 1.

One trap in the export format is worth naming. Each message carries both
``date`` ("2023-10-15T10:00:00", no zone, written in whatever timezone the
machine running the export happened to be in) and ``date_unixtime``, which is
unambiguous UTC. Trusting ``date`` silently shifts the whole chat by the
exporter's offset -- and since messages anchor shots by proximity in time,
that quietly mis-attributes the narration. ``date_unixtime`` wins wherever it
is present.
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[\wЀ-ӿ][-\wЀ-ӿ']*", re.UNICODE)
EMOJI_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

MEDIA_KEYS = ("photo", "file", "thumbnail")


@dataclass
class Message:
    msg_id: str
    ts_utc: datetime
    author: str | None
    text: str
    media_path: str | None = None
    media_type: str | None = None
    reply_to: str | None = None
    phase: str | None = None
    is_notable: bool = False
    usable_as_card: bool = False
    duration_s: float | None = None

    @property
    def has_media(self) -> bool:
        return self.media_path is not None


def flatten_text(value: Any) -> str:
    """Telegram stores text as a string, or a list mixing plain strings with
    entity objects ({'type': 'link', 'text': ...}). Both must collapse to one
    readable line."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text", "")).strip()
    if isinstance(value, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in value
        ).strip()
    return str(value).strip()


def message_time(raw: dict[str, Any]) -> datetime | None:
    """UTC timestamp, preferring the unambiguous unix field."""
    unix = raw.get("date_unixtime") or raw.get("date_unix")
    if unix is not None:
        try:
            return datetime.fromtimestamp(int(unix), tz=timezone.utc)
        except (ValueError, OSError, OverflowError):
            pass
    date = raw.get("date")
    if not date:
        return None
    try:
        dt = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_export(data: dict[str, Any] | str | Path) -> list[Message]:
    """Read ``result.json`` into Message rows, skipping service events."""
    if isinstance(data, (str, Path)):
        data = json.loads(Path(data).read_text(encoding="utf-8"))

    raw_messages = data.get("messages") or []
    out: list[Message] = []
    skipped = 0
    for raw in raw_messages:
        if raw.get("type") != "message":
            skipped += 1                       # joins, pins, calls
            continue
        ts = message_time(raw)
        if ts is None:
            skipped += 1
            continue
        text = flatten_text(raw.get("text"))
        if not text and raw.get("text_entities"):
            text = flatten_text(raw["text_entities"])

        media = next((raw[k] for k in MEDIA_KEYS if raw.get(k)), None)
        if isinstance(media, dict):
            media = media.get("file") or media.get("path")

        out.append(Message(
            msg_id=str(raw.get("id")),
            ts_utc=ts,
            author=raw.get("from"),
            text=text,
            media_path=str(media) if media else None,
            media_type=raw.get("media_type"),
            reply_to=str(raw["reply_to_message_id"]) if raw.get("reply_to_message_id") else None,
            duration_s=_f(raw.get("duration_seconds")),
        ))
    out.sort(key=lambda m: m.ts_utc)
    log.info("S02.5 parsed %d messages (%d service/undated skipped)", len(out), skipped)
    return out


def _f(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# -- classification ----------------------------------------------------

def classify_phase(ts: datetime, track_start: datetime | None,
                   track_end: datetime | None) -> str:
    """planning before the track, trek within it, after once it ends."""
    if track_start is None or track_end is None:
        return "planning"
    if ts < track_start:
        return "planning"
    if ts > track_end:
        return "after"
    return "trek"


def mark_notable(messages: Sequence[Message], *,
                 climb_events: Sequence[datetime] = (),
                 window_s: float = 1200.0,
                 long_factor: float = 3.0) -> None:
    """Flag messages worth surfacing, per spec S02.5.

    Notable if any of: it sits within ``window_s`` of a local maximum in
    altitude gain, it carries media, or it is unusually long against the
    thread median.
    """
    lengths = [len(m.text) for m in messages if m.text]
    median_len = statistics.median(lengths) if lengths else 0.0
    threshold = median_len * long_factor

    events = sorted(climb_events)
    for m in messages:
        near_climb = any(abs((m.ts_utc - e).total_seconds()) <= window_s for e in events)
        m.is_notable = bool(near_climb or m.has_media
                            or (threshold and len(m.text) > threshold))


def is_card_candidate(text: str, *, max_chars: int = 180, min_chars: int = 8,
                      min_words: int = 2) -> bool:
    """Short, self-contained and quotable -- Act 1's on-screen text.

    Rejects links, bare emoji and single words. These end up burned into the
    film, so the bar is "reads as a line someone said", not "is short".
    """
    t = (text or "").strip()
    if not (min_chars <= len(t) <= max_chars):
        return False
    if URL_RE.search(t):
        return False
    if EMOJI_ONLY_RE.match(t):
        return False
    words = WORD_RE.findall(t)
    if len(words) < min_words:
        return False
    # a fragment that is only a mention or a bare number is not a line
    if all(w.isdigit() for w in words):
        return False
    return True


def mark_cards(messages: Iterable[Message], *, max_chars: int = 180) -> int:
    n = 0
    for m in messages:
        m.usable_as_card = is_card_candidate(m.text, max_chars=max_chars)
        n += int(m.usable_as_card)
    return n


# -- vocabulary --------------------------------------------------------

# Deliberately small: this is not a language model, it is a stop list for
# extracting proper nouns and recurring gear/joke terms out of chat.
STOPWORDS = {
    "the", "and", "for", "you", "that", "this", "with", "have", "not", "but",
    "все", "как", "что", "это", "так", "там", "тут", "уже", "если", "меня",
    "тебя", "надо", "нас", "они", "мне", "его", "ещё", "еще", "быть", "было",
    "просто", "очень", "когда", "потом", "будет", "можно", "нужно", "давай",
}


def extract_vocabulary(messages: Sequence[Message], *, top_n: int = 60,
                       min_count: int = 3) -> list[str]:
    """Proper nouns and recurring terms, for injection into S04's caption prompt.

    Without this the captions come back in generic model English -- "two men
    walking on a mountain path" instead of "Keller and Kulikov above Namche" --
    and retrieval in the operator's own language stops working.
    """
    proper = Counter()
    general = Counter()
    for m in messages:
        if not m.text:
            continue
        for i, w in enumerate(WORD_RE.findall(m.text)):
            low = w.lower()
            if len(w) < 3 or low in STOPWORDS or w.isdigit():
                continue
            # a capitalised word that is not merely sentence-initial
            if w[0].isupper() and i > 0:
                proper[w] += 1
            general[low] += 1

    terms = [w for w, c in proper.most_common(top_n) if c >= min_count]
    for w, c in general.most_common(top_n * 3):
        if len(terms) >= top_n:
            break
        if c >= min_count * 2 and w not in {t.lower() for t in terms}:
            terms.append(w)
    return terms[:top_n]


def resolve_media_path(media_path: str, chat_root: Path) -> Path | None:
    """Map an export-relative media reference onto a real file."""
    if not media_path:
        return None
    cleaned = media_path.replace("\\", "/").lstrip("./")
    candidate = chat_root / cleaned
    if candidate.exists():
        return candidate
    # exports occasionally reference by basename only
    name = Path(cleaned).name
    for sub in ("photos", "video_files", "round_video_messages", "files",
                "voice_messages", "stickers"):
        alt = chat_root / sub / name
        if alt.exists():
            return alt
    return None
