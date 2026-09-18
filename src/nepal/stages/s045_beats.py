"""S04.5 -- the beat sheet (Film v2 section 4.2).

One call. Claude reads every real transcript with its timing, the whole
chat anonymised, and the day table, and returns the spoken moments that
carry the story, the Act 1 quotes, the closing line, the pairs and the
trailer picks. The answer is validated against the rules of section 4.2;
if it breaks one, the complaints go back once and the corrected sheet is
validated again. What passes is written to `story_beats`, to
`work/beats/beats.json`, and to a Gate 2 page the operator reads.

The money is guarded twice before a token is bought: the prompt is counted
and refused above `beats.max_input_tokens` or `beats.estimate_usd_cap`, and
the project ledger refuses when the ceiling would be crossed. `--dry-run`
writes the prompt and the estimate and calls nothing, which is how the
prompt is read before it is paid for.
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from nepal import db, freshness
from nepal.cloud import claude as claude_mod, spend
from nepal.config import Config
from nepal.story import beats_input, beats_schema

log = logging.getLogger(__name__)
STAGE = "S04.5"


class BeatsRefused(RuntimeError):
    """Refused before the call: the prompt is too big or too dear."""


class BeatsInvalid(RuntimeError):
    """The sheet still broke the rules after the retries; nothing was written."""

    def __init__(self, errors: Sequence[str], path: Path):
        self.errors = list(errors)
        self.path = path
        super().__init__(f"{len(errors)} rule(s) still broken after the retries; "
                         f"the last answer is at {path}")


def _client(cfg: Config):
    model = str(cfg.get("beats.model"))
    price = claude_mod.Price.from_cfg(cfg, model)
    return claude_mod.Claude(model, price=price, effort=str(cfg.get("beats.effort", "high")),
                             max_tokens=int(cfg.get("beats.max_output_tokens", 16000)))


def count_input(claude, prompt: beats_input.Prompt, *, chars_per_token: float) -> tuple[int, str]:
    """Tokens in the prompt: counted by the API when it can be, estimated
    from characters when it cannot (no SDK or no key -- the dry run on a
    laptop). Says which."""
    try:
        return claude.count_tokens(prompt.system, prompt.messages()), "counted"
    except Exception as exc:                       # ImportError, auth, network
        log.info("S04.5 could not count tokens (%s); estimating from characters", exc)
        return int((len(prompt.system) + len(prompt.user)) / chars_per_token), "estimated"


def retry_messages(prompt: beats_input.Prompt, answer_text: str,
                   errors: Sequence[str]) -> list[dict[str, Any]]:
    """The conversation for the second try: the answer, and what was wrong."""
    complaint = ("The validator found these problems with your beat sheet:\n\n"
                 + "\n".join(f"- {e}" for e in errors)
                 + "\n\nReturn the complete corrected JSON object -- every field, not "
                   "only the changed ones -- following the same rules.")
    return prompt.messages() + [{"role": "assistant", "content": answer_text},
                                {"role": "user", "content": complaint}]


def resolve_handles(doc: Mapping[str, Any],
                    shot_info: Mapping[str, Mapping[str, Any]]) -> None:
    """Put the real shot ids back where the model wrote handles.

    The model only ever saw ``S0001``-style handles (the recording ids carry
    surnames). Once the sheet is valid, every speech beat gets its shot id
    back and keeps the handle beside it, so the rows, the JSON and the page
    all name the recording and the input can still be traced.
    """
    for b in doc.get("beats") or []:
        if b.get("kind") == "speech" and b.get("shot_id") in shot_info:
            b["handle"] = b["shot_id"]
            b["shot_id"] = shot_info[b["shot_id"]]["shot_id"]


def rows_from_doc(doc: Mapping[str, Any], *, created_utc: str) -> list[dict[str, Any]]:
    """story_beats rows: every column on every row (db.upsert insists)."""
    def row(**kw: Any) -> dict[str, Any]:
        base = {"beat_id": None, "kind": None, "act": None, "shot_id": None, "msg_id": None,
                "src_in": None, "src_out": None, "text": None, "levity": 0, "effect": None,
                "rationale": None, "rank": None, "created_utc": created_utc}
        base.update(kw)
        return base
    rows = [row(beat_id=str(b["beat_id"]), kind=str(b["kind"]), act=b.get("act"),
                shot_id=b.get("shot_id"), msg_id=b.get("msg_id"),
                src_in=b.get("src_in"), src_out=b.get("src_out"), text=b.get("text"),
                levity=int(bool(b.get("levity"))), effect=b.get("effect"),
                rationale=b.get("rationale"), rank=b.get("rank"))
            for b in doc.get("beats") or []]
    closing = doc.get("closing") or {}
    if closing.get("msg_id"):
        rows.append(row(beat_id="closing", kind="closing", msg_id=closing["msg_id"],
                        text=closing.get("text")))
    if doc.get("title"):
        rows.append(row(beat_id="title", kind="title", text=str(doc["title"])))
    return rows


def render_gate2(doc: Mapping[str, Any], meta: Mapping[str, Any],
                 shot_info: Mapping[str, Mapping[str, Any]],
                 msg_info: Mapping[str, Mapping[str, Any]]) -> str:
    """Gate 2, as a page: the beats by act with their reasons, a link to the
    proxy at the moment, the quotes, the pairs, the closing line. No name
    anywhere, because none was given to the model."""
    e = html.escape
    beats = list(doc.get("beats") or [])
    speech = [b for b in beats if b.get("kind") == "speech"]
    quotes = [b for b in beats if b.get("kind") == "quote"]
    acts = sorted({int(b["act"]) for b in speech if b.get("act") is not None})
    out = [f"<!doctype html><meta charset='utf-8'><title>Gate 2 — {e(str(doc.get('title', '')))}</title>",
           "<style>body{font:15px/1.45 system-ui,sans-serif;max-width:60em;margin:2em auto;"
           "padding:0 1em;color:#222}h2{margin-top:2em}.beat{margin:.8em 0;padding:.6em .9em;"
           "border-left:4px solid #999;background:#fafafa}.beat.levity{border-color:#e6a100}"
           ".t{font-size:1.1em}.why{color:#555}.meta{color:#777;font-size:.9em}"
           "code{font-size:.9em}</style>",
           f"<h1>Gate 2 — the beat sheet</h1><p class='t'><b>{e(str(doc.get('title', '')))}</b></p>",
           f"<p class='meta'>{meta.get('n_shots', 0)} transcripts, {meta.get('n_messages', 0)} "
           f"messages read; {len(speech)} speech beats, {len(quotes)} quotes; "
           f"model {e(str(meta.get('model', '')))}, {meta.get('usd', 0):.2f} USD, "
           f"{meta.get('attempts', 1)} attempt(s).</p>",
           "<p>Drop, retime or re-rank here means: edit <code>work/beats/beats.json</code> "
           "and re-run <code>nepal beats --force</code>, or hand-edit <code>story_beats</code>. "
           "Times are seconds on the recording's clock; the link opens the proxy there.</p>"]
    for act in acts:
        out.append(f"<h2>Act {act}</h2>")
        for b in sorted((b for b in speech if int(b['act']) == act),
                        key=lambda b: (shot_info.get(b.get('handle') or b['shot_id'], {})
                                       .get('utc', ''), float(b.get('src_in') or 0))):
            rid = str(b.get("shot_id", "")).split("#")[0]
            href = f"../../proxies/{e(rid)}_eq.mp4#t={float(b.get('src_in') or 0):.1f}"
            cls = "beat levity" if b.get("levity") else "beat"
            out.append(
                f"<div class='{cls}'><div class='t'>{e(str(b.get('text', '')))}</div>"
                f"<div class='meta'>#{b.get('rank')} · <code>{e(str(b.get('beat_id')))}</code> · "
                f"<a href='{href}'>{e(str(b.get('shot_id')))}</a> "
                f"{float(b.get('src_in') or 0):.1f}–{float(b.get('src_out') or 0):.1f} s"
                f" · effect {e(str(b.get('effect')))}"
                f"{' · levity' if b.get('levity') else ''}</div>"
                f"<div class='why'>{e(str(b.get('rationale', '')))}</div></div>")
    out.append("<h2>Act 1 — the chat</h2>")
    for b in sorted(quotes, key=lambda b: int(b.get("rank") or 99)):
        out.append(f"<div class='beat'><div class='t'>{e(str(b.get('text', '')))}</div>"
                   f"<div class='meta'>#{b.get('rank')} · <code>{e(str(b.get('beat_id')))}</code>"
                   f" · message {e(str(b.get('msg_id')))}</div>"
                   f"<div class='why'>{e(str(b.get('rationale', '')))}</div></div>")
    closing = doc.get("closing") or {}
    out.append(f"<h2>The last words</h2><div class='beat'><div class='t'>"
               f"{e(str(closing.get('text', '')))}</div><div class='meta'>message "
               f"{e(str(closing.get('msg_id', '')))}</div></div>")
    pairs = doc.get("pairs") or []
    if pairs:
        out.append("<h2>Planning versus reality</h2>")
        by_id = {str(b.get("beat_id")): b for b in beats}
        for p in pairs:
            msg = msg_info.get(p.get("planning_msg_id") or "", {})
            beat = by_id.get(str(p.get("trek_beat_id")), {})
            out.append(f"<div class='beat'><div>“{e(str(msg.get('text', '')))}”</div>"
                       f"<div class='meta'>… over <code>{e(str(p.get('trek_beat_id')))}</code>: "
                       f"{e(str(beat.get('text', '')))}</div>"
                       f"<div class='why'>{e(str(p.get('why', '')))}</div></div>")
    notes = doc.get("act_notes") or []
    if notes:
        out.append("<h2>Notes for the assembly</h2><ul>")
        out += [f"<li>Act {e(str(n.get('act')))}: {e(str(n.get('note', '')))}</li>" for n in notes]
        out.append("</ul>")
    ideas = doc.get("stat_card_ideas") or []
    if ideas:
        out.append("<h2>Statistics worth a card</h2><ul>")
        out += [f"<li>{e(str(i))}</li>" for i in ideas]
        out.append("</ul>")
    tr = doc.get("trailer") or {}
    if tr:
        out.append(f"<h2>The trailer</h2><p>Hook <code>{e(str(tr.get('hook_beat_id')))}</code>, "
                   f"cliffhanger <code>{e(str(tr.get('cliffhanger_beat_id')))}</code></p><ul>")
        out += [f"<li>{e(str(c))}</li>" for c in tr.get("cards") or []]
        out.append("</ul>")
    return "\n".join(out) + "\n"


def run(cfg: Config, *, force: bool = False, dry_run: bool = False,
        claude=None) -> dict[str, Any]:
    conn = db.init(cfg.db_path)
    report: dict[str, Any] = {"stage": STAGE, "started_utc": db.utcnow()}
    report["skipped_stale"] = freshness.warn_if_stale(
        log, conn, STAGE, force=force, rerun_hint="nepal beats --force")
    have = conn.execute("SELECT COUNT(*) FROM story_beats").fetchone()[0]
    if have and not force and not dry_run:
        # A dollar a run and an answer that should be stable: re-run only
        # when asked, like the spec says.
        log.info("S04.5 story_beats holds %d row(s); not calling again without --force", have)
        report["skipped"] = f"{have} beats exist; use --force"
        conn.close()
        return report

    rules = beats_schema.Rules.from_cfg(cfg)
    prompt = beats_input.build_prompt(cfg, conn, rules=rules.as_prompt_dict())
    out_dir = cfg.workdir("beats")
    (out_dir / "prompt.txt").write_text(prompt.system + "\n\n" + "=" * 72 + "\n\n" + prompt.user)
    public_meta = {k: v for k, v in prompt.meta.items() if k not in ("shot_info", "msg_info")}
    log.info("S04.5 prompt: %d transcripts (%d segments, %d whole-shot only), %d messages, "
             "%d days, %d characters -> %s", prompt.meta["n_shots"], prompt.meta["n_segments"],
             prompt.meta["n_synthetic"], prompt.meta["n_messages"], prompt.meta["n_days"],
             prompt.meta["chars"], out_dir / "prompt.txt")

    claude = claude or _client(cfg)
    n_in, how = count_input(claude, prompt,
                            chars_per_token=float(cfg.get("beats.estimate_chars_per_token", 3.0)))
    max_out = int(cfg.get("beats.max_output_tokens", 16000))
    estimate = claude_mod.estimate_usd(n_in, max_out, claude.price)
    ledger = spend.ledger(cfg)
    report.update({"input_tokens": n_in, "tokens_how": how, "estimate_usd": estimate,
                   "ledger_usd": round(ledger.total(), 4), "prompt": public_meta})
    log.info("S04.5 %d input tokens (%s), at most %d out: estimate %.2f USD; ledger %.2f "
             "of %s", n_in, how, max_out, estimate, ledger.total(),
             cfg.get("cloud.spend_ceiling_usd"))
    if n_in > int(cfg.get("beats.max_input_tokens", 200000)):
        raise BeatsRefused(f"{n_in} input tokens is over beats.max_input_tokens; "
                           f"the prompt is at {out_dir / 'prompt.txt'}")
    if estimate > float(cfg.get("beats.estimate_usd_cap", 3.0)):
        raise BeatsRefused(f"estimated {estimate:.2f} USD is over beats.estimate_usd_cap")
    ledger.guard(estimate, ceiling_usd=float(cfg.get("cloud.spend_ceiling_usd")))
    if dry_run:
        report["dry_run"] = True
        (out_dir / "prompt_meta.json").write_text(json.dumps(report, indent=2, default=str))
        conn.close()
        return report

    messages = prompt.messages()
    attempts = 1 + int(cfg.get("beats.retries", 1))
    completions: list[claude_mod.Completion] = []
    errors: list[str] = []
    doc: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        log.info("S04.5 calling %s (attempt %d of %d)", claude.model, attempt, attempts)
        try:
            comp = claude.complete_json(prompt.system, messages, beats_schema.OUTPUT_SCHEMA)
        except claude_mod.ClaudeError as exc:
            # Paid for all the same: the ledger says so, and what did arrive
            # is kept to be read. Then the failure is the caller's.
            ledger.record("beats", exc.usd,
                          detail=f"{claude.model} attempt {attempt} {type(exc).__name__}: "
                                 f"{exc.input_tokens} in, {exc.output_tokens} out")
            (out_dir / f"response_{attempt}_failed.txt").write_text(exc.text or "")
            conn.close()
            raise
        completions.append(comp)
        ledger.record("beats", comp.usd,
                      detail=f"{comp.model} attempt {attempt}: {comp.input_tokens} in, "
                             f"{comp.output_tokens} out")
        (out_dir / f"response_{attempt}.json").write_text(comp.text)
        doc = comp.data
        errors = beats_schema.validate(doc, shot_info=prompt.meta["shot_info"],
                                       msg_info=prompt.meta["msg_info"], rules=rules)
        log.info("S04.5 attempt %d: %d in, %d out, %.2f USD, %d rule(s) broken",
                 attempt, comp.input_tokens, comp.output_tokens, comp.usd, len(errors))
        for err in errors[:12]:
            log.info("S04.5   - %s", err)
        if not errors:
            break
        messages = retry_messages(prompt, comp.text, errors)
    usd = round(sum(c.usd for c in completions), 4)
    report.update({"attempts": len(completions), "usd": usd,
                   "input_tokens_actual": completions[-1].input_tokens,
                   "output_tokens": completions[-1].output_tokens, "errors": errors})
    if errors or doc is None:
        conn.close()
        raise BeatsInvalid(errors, out_dir / f"response_{len(completions)}.json")

    created = db.utcnow()
    resolve_handles(doc, prompt.meta["shot_info"])
    rows = rows_from_doc(doc, created_utc=created)
    conn.execute("DELETE FROM story_beats")
    db.upsert(conn, "story_beats", ["beat_id"], rows)
    meta = {**public_meta, "model": completions[-1].model, "usd": usd,
            "attempts": len(completions), "created_utc": created}
    (out_dir / "beats.json").write_text(json.dumps({"meta": meta, "sheet": doc},
                                                   ensure_ascii=False, indent=1))
    page = cfg.workdir("gates", "gate2") / "index.html"
    page.write_text(render_gate2(doc, meta, prompt.meta["shot_info"], prompt.meta["msg_info"]))
    speech = [b for b in doc.get("beats") or [] if b.get("kind") == "speech"]
    per_act: dict[str, int] = {}
    for b in speech:
        per_act[str(b.get("act"))] = per_act.get(str(b.get("act")), 0) + 1
    report.update({"title": doc.get("title"), "n_rows": len(rows), "n_speech": len(speech),
                   "n_quotes": sum(1 for b in doc.get("beats") or [] if b.get("kind") == "quote"),
                   "n_pairs": len(doc.get("pairs") or []), "per_act": per_act,
                   "beats_json": str(out_dir / "beats.json"), "gate2": str(page)})
    log.info("S04.5 %r: %d speech beats %s, %d quotes, %d pairs; %.2f USD; %s",
             doc.get("title"), len(speech), per_act, report["n_quotes"], report["n_pairs"],
             usd, page)
    db.mark_unit(conn, STAGE, "beats", detail=json.dumps(report, default=str)[:2000])
    report["finished_utc"] = db.utcnow()
    cfg.work("reports", "s045_beats.json").write_text(json.dumps(report, indent=2, default=str))
    conn.close()
    return report
