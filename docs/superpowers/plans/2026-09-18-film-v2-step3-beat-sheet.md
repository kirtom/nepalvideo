# Film v2 — Step 3: the voice spine (hallucination filter, segment times, beat sheet) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every transcript carries its segment times and whisper's confidence in the database; a deterministic filter marks the hallucinated ones; one Claude call reads every real transcript, the anonymised chat and the day table and returns the beat sheet of §4.2 — the 10–16 spoken moments, the Act 1 quotes, the closing line, the pairs and the trailer picks — validated, written to `story_beats` and `work/beats/beats.json`, and shown on a Gate 2 page. `speech_first` goes.

**Architecture:** `process/asr.py` keeps whisper's per-segment confidence and word times; `process/hallucination.py` is a pure filter applied in S03 beside the gate; `story/brief.py`, `story/beats_input.py`, `story/beats_schema.py` are pure (prompt assembly, anonymisation, validation); `cloud/claude.py` is the one thin wrapper around the Anthropic SDK with the spend ledger in front of it and a fake for tests; `stages/s045_beats.py` is the stage, reached as `nepal beats`. Runs on the box: `nepal remote run beats`.

**Tech Stack:** Python 3.10+, `anthropic>=1.6` (SDK, `claude-opus-5`, adaptive thinking, structured output via `output_config.format`, streaming with `get_final_message`), faster-whisper (word timestamps), SQLite, pytest with recorded responses.

**Spec:** `docs/superpowers/specs/2026-09-16-film-v2-voice-spine-design.md` §3.1, §3.2, §4, §8.6, §8.7, §8.12, §9 (Gate 2), §10 (contract tests), §11 step 3.

## Global Constraints

- **Nothing computes on the local machine.** There is no venv in the worktree and no pytest on the system Python. Pure single-file tests may be run through the main checkout's interpreter with `PYTHONPATH=src /data/projects/nepalvideo/.venv/bin/python -m pytest -q tests/<one file>` (sub-second); the full suite, the re-transcription and the beat sheet run on the box.
- Every tunable lives in `config/pipeline.yaml` under three new top-level blocks `asr:`, `beats:`, `api:`; duplicate top-level keys raise, so `grep -n '^asr:\|^beats:\|^api:'` must find nothing first.
- Credentials never enter the repo. The API key reaches the box only as instance metadata `anthropic-api-key` → `ANTHROPIC_API_KEY` in `~/.profile` (bootstrap, step 2). The wrapper reads the environment through the SDK's default client; it never takes a key argument from config.
- Spend: `beats.estimate_usd_cap` refuses a call estimated above it; the ledger guard refuses when the ceiling would be crossed; the actual cost is recorded after the call from `usage`.
- No author names anywhere in the prompt or in any output: speakers are `A`, `B`, `C` by order of first appearance. The test asserts the names are absent.
- **Table name:** the spec's `beats` table collides with the music beat grid S02.7 writes and S05 reads (`beats(track_id, t_s, is_downbeat)`). The voice-spine table is `story_beats`; §3.2 of the spec gets a one-line amendment. Renaming a table two stages read is not worth the churn.
- Commit after every task, message ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013XaQhkq3CDL5Qy4AUKcX5M`.

**Facts fixed on 2026-09-18:** 1,060 video shots, 516 with speech, 498 with a transcript; 332 transcript JSON files exist (the rest were transcribed before the JSON was written — they get one synthetic segment spanning the shot until re-transcribed). Whisper wrote one segment per shot in most cases, so cut points inside a shot need word timestamps: the re-transcription with `word_timestamps=True` runs on the box (~0.7× realtime on 8 cores, about 1.5 h for 2 h of speech) detached, while the rest of the step is built. Messages: 1,936 in phases `planning` (1,582) and `after` (354), none during the trek; three authors. Face clusters: `keller`, `kulikov`; the third cluster (Dharma, the guide) is unlabelled. The box has no `anthropic-api-key` metadata yet: the live call waits on the operator; everything up to `nepal beats --dry-run` (prompt written, tokens counted, no call) is verifiable without it.

## File structure

| File | Responsibility |
|---|---|
| `config/pipeline.yaml` | `asr:` (filter thresholds, phrases), `beats:` (model, counts, caps), `api:` (prices) |
| `src/nepal/db.py` | `shots.transcript_json`, `shots.hallucinated` migrations; `story_beats` table |
| `src/nepal/process/asr.py` | segments keep `no_speech_prob`, `compression_ratio`, `avg_logprob`, `words`; `word_timestamps=True` |
| `src/nepal/process/hallucination.py` (new) | pure: `segment_reasons`, `shot_hallucinated`, `repeated_ngram` |
| `src/nepal/stages/s03_process.py` | S03.5 writes `transcript_json`; new sub-step `hallucination` (backfill + filter); gate zeroes `has_speech` on hallucinated shots |
| `src/nepal/cloud/claude.py` (new) | `Claude` (thin SDK wrapper: `count_tokens`, `complete_json`), `FakeClaude`, `estimate_usd`, `usage_usd` |
| `src/nepal/story/__init__.py`, `brief.py` (new) | the brief and the rules, as the prompt text |
| `src/nepal/story/beats_input.py` (new) | pure: `anonymise`, `transcript_rows`, `chat_rows`, `day_rows`, `build_messages` |
| `src/nepal/story/beats_schema.py` (new) | pure: `OUTPUT_SCHEMA`, `Rules.from_cfg`, `validate` |
| `src/nepal/stages/s045_beats.py` (new) | the stage: gather → prompt → call → validate (retry once) → write rows, JSON, Gate 2 page |
| `src/nepal/cli.py` | `nepal beats [--force] [--dry-run]` |
| `src/nepal/process/assemble.py`, `src/nepal/stages/s05_cut.py` | `speech_first` removed |
| `src/nepal/cloud/sync.py` | `beats` already in PULL; nothing to add |
| `tools/cloud/jobs-step3.sh` (new) | tests, `s03 --redo asr` detached, then `beats --dry-run` |
| `tests/test_asr.py`, `test_hallucination.py`, `test_cloud_claude.py`, `test_beats_input.py`, `test_beats_schema.py`, `test_s045_beats.py`, `test_db.py`, `test_assemble.py` | one file per unit |
| `docs/STATE.md`, `README.md`, spec §3.2, `CLAUDE.md` | what step 3 produced; the `nepal beats` command; the table name; the trap about whisper's one-segment shots |

---

### Task 1: Config blocks, the two shot columns and `story_beats`

- Append `asr:`, `beats:`, `api:` to `config/pipeline.yaml`.
- `db.MIGRATIONS` += `("shots", "transcript_json", "TEXT")`, `("shots", "hallucinated", "INTEGER")`; `SCHEMA` += `story_beats` per §3.2 (`beat_id` PK, `kind`, `act`, `shot_id`, `msg_id`, `src_in`, `src_out`, `text`, `levity`, `effect`, `rationale`, `rank`, plus `created_utc`).
- Tests: `test_db.py` — a fresh DB has the columns and the table; an old DB gains them.

### Task 2: Whisper keeps its confidence and its word times

- `asr.join_segments` keeps `no_speech_prob`, `compression_ratio`, `avg_logprob` when the segment has them and `words: [{start_s, end_s, word, probability}]` when present; absent attributes stay absent (older JSON stays readable).
- `asr.transcribe_window(..., word_timestamps=True)` passes the flag through.
- `asr.synthetic_transcript(shot_row) -> dict` gives a one-segment transcript spanning the shot for rows that only have text.
- `s03_process.transcribe_shots` writes `transcript_json` beside `transcript`; `s03_process.backfill_transcript_json(cfg, conn)` fills NULLs from the JSON files or the synthetic segment (idempotent, counts what it did).
- Tests: `test_asr.py` — confidence and words carried and offset; absent fields absent; synthetic segment spans the shot; `test_e2e_s03`-style backfill on a tmp DB.

### Task 3: The hallucination filter

- `process/hallucination.py`: `repeated_ngram(text, n=3, min_repeats=3) -> str | None`; `segment_reasons(seg, *, phrases, max_no_speech_prob, max_compression_ratio) -> list[str]`; `shot_hallucinated(segments, *, speech_s, speech_min_s, ...) -> tuple[bool, dict]` (all segments flagged, or no VAD speech). Phrase match is case-insensitive substring.
- S03 sub-step `hallucination` (always runs, after `asr`, before `faces`): backfill, then set `hallucinated` per shot and store per-segment reasons back into `transcript_json` (`segments[i].hallucinated_reasons`), report counts.
- `apply_gate`: `has_speech = 0` where `hallucinated = 1` (so the reprieve and the context term never see a hallucination).
- Tests: `test_hallucination.py` — each rule; a real line survives; the gate test.

### Task 4: The Claude wrapper with the ledger in front

- `cloud/claude.py`: `Price(input, output)` from `api.prices_usd_per_mtok`; `estimate_usd(n_in, n_out, price)`; `usage_usd(usage, price)`; class `Claude(model, *, effort, max_tokens)` with `count_tokens(system, messages) -> int` and `complete_json(system, messages, schema) -> Completion(text, data, usage, stop_reason, usd)` using `client.messages.stream(..., output_config={"format": {"type": "json_schema", "schema": ...}, "effort": ...})` + `get_final_message()`; `refusal` and `max_tokens` stop reasons raise `ClaudeRefused` / `ClaudeTruncated`. `FakeClaude(responses: list[str], n_tokens=1000)` records calls and returns canned text with a fixed usage.
- `guarded_call(ledger, ceiling, estimate) ` is the stage's job (Task 6); the wrapper does not know the ledger.
- `pyproject.toml`: new extra `api = ["anthropic>=1.6"]`; `cloud/remote.py EXTRAS` gains `api`; bootstrap's pip line too.
- Tests: `test_cloud_claude.py` — prices, estimate arithmetic, the fake records what it was asked, `Claude` imports lazily (no SDK needed to import the module).

### Task 5: Prompt assembly (pure) and the validator (pure)

- `story/brief.py`: `BRIEF` (tone, protagonists, the six acts with musical character), `RULES(rules)` renders the numeric rules from config, `OUTPUT_HINT`.
- `story/beats_input.py`: `anonymise(authors) -> dict[name, tag]`; `transcript_rows(conn) -> list[dict]` (non-hallucinated shots with a transcript, any status, with `gate` flag, day, place, alt, source, face cluster label mapped to `A`/`B`/`guide`, motion class, segments with `sid`, `start_s`, `end_s`, `text`, and word cut points where present); `chat_rows(conn)`; `day_rows(conn, cfg)`; `render_*` to compact text blocks; `build_messages(cfg, conn) -> (system, user_text, meta)`; `cut_points(segments) -> sorted list of floats` (segment edges plus word edges).
- `story/beats_schema.py`: `OUTPUT_SCHEMA` (JSON schema, `additionalProperties: false`, beats with `beat_id`); `Rules.from_cfg(cfg)`; `validate(doc, *, cut_points_by_shot, msg_ids, planning_msg_ids, act_of_shot, rules) -> list[str]` enforcing §4.2: counts, chronological by `src_in` within shot order, ≥1 per act ≥2 and ≥1 levity per act ≥2, max length, `src_in`/`src_out` on cut points (±0.05 s) and `src_in < src_out`, quotes count and length, closing from the `after` phase and ≤ max chars, Act 4 ≤ 1 beat, pairs reference planning messages and existing beat ids, trailer ids exist, text of a speech beat is a substring-ish of the transcript (normalised), no unknown ids.
- Tests: `test_beats_input.py` (no names; A/B by first appearance; segments carry ids and cut points; day rows have km/gain from Strava), `test_beats_schema.py` (one test per rule, a good sheet passes).

### Task 6: The stage and the command

- `stages/s045_beats.py`: `run(cfg, *, force, dry_run)`: skip when `story_beats` has rows and not force; `build_messages`; write `work/beats/prompt.txt`; count tokens (via `Claude.count_tokens`, or estimate from characters in dry-run without a key) → estimate → `ledger.guard`; refuse above `beats.estimate_usd_cap` or `beats.max_input_tokens`; call; validate; on errors append an assistant/user turn with the error list and call once more; on success write rows (`DELETE FROM story_beats` first), `work/beats/beats.json` (the raw doc plus meta: model, usage, usd, validated_at), `work/beats/response_raw.json`, and `work/gates/gate2/index.html` (static: title, beats per act with time, text, rationale, levity, effect; quotes; closing; pairs; act notes; stat ideas; trailer; a `file://` link to the proxy at `#t=src_in`); record the ledger entry.
- `nepal beats [--force] [--dry-run]` in `cli.py`; `s05_cut.build_timeline` no longer calls `speech_first` (candidates sorted by score); `assemble.speech_first` deleted with its test.
- Tests: `test_s045_beats.py` — end to end on a tmp DB with `FakeClaude`: a bad first answer then a good one → rows written, ledger has one entry, Gate 2 page has no author name; dry-run writes the prompt and calls nothing; `test_assemble.py` — `speech_first` gone.

### Task 7: On the box

- `tools/cloud/jobs-step3.sh`: tests (whole suite), `nepal s03 --redo asr` (re-transcription with word timestamps; the filter and gate follow inside S03), `nepal beats --dry-run`, push. Launched detached; polled every minute; the log read as it grows.
- Docs: STATE.md step 3 section (numbers from the run), README (`nepal beats`, Gate 2), spec §3.2 amendment, CLAUDE.md trap ("whisper returns one segment per 20 s shot unless asked for words").
- Then: `needs input` — the operator sets `anthropic-api-key` on the instance, and `nepal remote run beats` makes the live call (about a dollar).
