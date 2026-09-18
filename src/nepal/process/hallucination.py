"""S03.5b -- the hallucination filter (Film v2 section 4.1).

Whisper invents text where there is nothing to hear. On this corpus it
invented subtitle credits ("Субтитры сделал DimaTorzok"), television
sign-offs ("Продолжение следует") and one phrase looped a dozen times, and
the first draft placed two of those first in their acts because speech was a
claim on a slot. The first cut of the film was, for a few seconds, a credit
for a subtitler who does not exist.

Every rule here is deterministic and reads what is already stored, so the
filter is re-run on every S03 pass in milliseconds and the transcripts are
never re-made for it. A shot is hallucinated when every segment it has is,
or when the VAD heard no speech in it at all.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_PHRASES = ("Субтитры сделал", "DimaTorzok", "Продолжение следует",
                   "Редактор субтитров", "Спасибо за просмотр", "Подписывайтесь")
MAX_NO_SPEECH_PROB = 0.6
MAX_COMPRESSION_RATIO = 2.4
NGRAM_N = 3
NGRAM_MIN_REPEATS = 3

_WORD = re.compile(r"\w+", re.UNICODE)


def repeated_ngram(text: str, *, n: int = NGRAM_N,
                   min_repeats: int = NGRAM_MIN_REPEATS) -> str | None:
    """The first n-gram that occurs at least ``min_repeats`` times, or None.

    A person repeats a word; a model in a loop repeats a phrase. Three words
    three times inside one segment is the loop, not emphasis.
    """
    words = [w.lower() for w in _WORD.findall(text or "")]
    if len(words) < n * min_repeats:
        return None
    seen: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        seen[g] = seen.get(g, 0) + 1
        if seen[g] >= min_repeats:
            return " ".join(g)
    return None


def segment_reasons(seg: Mapping[str, Any], *,
                    phrases: Iterable[str] = DEFAULT_PHRASES,
                    max_no_speech_prob: float = MAX_NO_SPEECH_PROB,
                    max_compression_ratio: float = MAX_COMPRESSION_RATIO,
                    ngram_n: int = NGRAM_N,
                    ngram_min_repeats: int = NGRAM_MIN_REPEATS) -> list[str]:
    """Why this segment is a hallucination; empty when it is speech.

    The confidence rules apply only where whisper's numbers were stored:
    a transcript from before they were kept is judged on its text alone
    rather than waved through or thrown out for a missing key.
    """
    text = seg.get("text") or ""
    low = text.lower()
    reasons: list[str] = []
    for p in phrases:
        if p and p.lower() in low:
            reasons.append(f"phrase:{p}")
            break
    g = repeated_ngram(text, n=ngram_n, min_repeats=ngram_min_repeats)
    if g:
        reasons.append(f"loop:{g}")
    nsp = seg.get("no_speech_prob")
    if nsp is not None and float(nsp) > max_no_speech_prob:
        reasons.append(f"no_speech_prob:{float(nsp):.2f}")
    cr = seg.get("compression_ratio")
    if cr is not None and float(cr) > max_compression_ratio:
        reasons.append(f"compression_ratio:{float(cr):.2f}")
    return reasons


def shot_hallucinated(segments: Sequence[Mapping[str, Any]], *, speech_s: float | None,
                      speech_min_s: float, **rule_kw: Any) -> tuple[bool, list[list[str]]]:
    """Whether the shot's transcript is a hallucination, and why per segment.

    True when every segment has a reason, or when the VAD measured less
    speech in the shot than the gate's floor -- text over a window with no
    voice in it is the textbook case. A shot with no segments at all is not
    hallucinated; it is silent, and has_speech says so already.
    """
    per_seg = [segment_reasons(s, **rule_kw) for s in segments]
    if speech_s is not None and float(speech_s) < float(speech_min_s):
        return True, [r + ["no_vad_speech"] for r in per_seg] if per_seg else []
    if not per_seg:
        return False, []
    return all(per_seg), per_seg


def rules_from_cfg(cfg) -> dict[str, Any]:
    """The rule thresholds, from config/pipeline.yaml's asr block."""
    return {"phrases": tuple(cfg.get("asr.hallucination_phrases", list(DEFAULT_PHRASES))),
            "max_no_speech_prob": float(cfg.get("asr.max_no_speech_prob", MAX_NO_SPEECH_PROB)),
            "max_compression_ratio": float(cfg.get("asr.max_compression_ratio",
                                                   MAX_COMPRESSION_RATIO)),
            "ngram_n": int(cfg.get("asr.ngram_n", NGRAM_N)),
            "ngram_min_repeats": int(cfg.get("asr.ngram_min_repeats", NGRAM_MIN_REPEATS))}
