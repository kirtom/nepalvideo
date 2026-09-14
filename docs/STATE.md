# Where the project stands

**As of 2026-09-14.** Keep this current: it is what a new session reads to
avoid re-deriving a fortnight of findings. When a stage lands or a number
changes, edit this file in the same commit.

## Build status

| Milestone | Scope | Status |
|---|---|---|
| 1 | S01 Probe + S02 Spine | **done**, validated end to end |
| 2 | S03 per-clip processing | **S03.0–.3 and .7 done**; .4 .5 .6 .8 remain |
| 5 | S04 Semantic + S05 Score | not started |
| 7 | S06 Assemble + S07 Draft render | not started |
| 3, 4, 6, 8, 9 | containerise, Batch, gates, conform, Step Functions | not started |

Everything so far runs locally. Nothing requires an AWS account yet, and the
first AWS action when one is needed is the Budgets alarm (spec §8.1).

## The last full run

S01 + S03.0/.1/.2 on the real corpus, 4.5 hours wall clock. These numbers are
the baseline to compare the next run against — but three changes since then
invalidate parts of them, noted inline.

- **S01** — 477 recordings (79 multi-chapter), **15 continuity notes** (down
  from 358 before frame-shape classification). Frame shapes: flat 69,
  single_fisheye 30, dual_fisheye 22. 69 files reclassified out of 360.
  302 capture-time conflicts resolved `CreationDate over CreateDate`, worst
  `IMG_2622.MOV` at 574 days.
- **S03.1** — 480 recordings, 477 built, 3 failed, 3:38:38. Modes: flat 455,
  dual_fisheye 22. `lens_pair` never fired, correctly: every single-lens
  `.insv` belongs to a recording that also holds a 1024×512 dual-fisheye
  `.lrv`, which is cheaper and carries the same seconds.
  *Invalidated:* this ran with `proxy_fps` silently unset (duplicate YAML key),
  so every proxy was encoded at source rate. The proxies are usable; a rebuild
  would be 2–4× faster but is not worth 3.5 hours.
  The 3 failures were stale recordings from the pre-fix grouping; S01.2 now
  deletes those.
- **S03.2** — 794 shots in 41:39. Per act {1:89, 2:102, 3:261, 4:138, 5:194,
  None:10}; 606 positioned; **395 recordings had no detected cut**.
  *Invalidated:* `process.max_shot_s` (20 s) now divides long scenes. Expect
  roughly 900–1000 video shots on the next run.
- **S03.0** — 623 photo shots from 651 dated photos in 8:46. Per act
  {1:11, 2:127, 3:262, 4:56, 5:167}; 28 below the gate; zero unreadable.
  *Invalidated:* stills were measured on raw Laplacian variance against
  log-scale thresholds, so the gate passed almost everything and `score_tech`
  saturated. Both measures now come from the S03.3 functions.

## Do these next, in order

1. **`pip install -e '.[vision,music,asr]'`** — the working install is
   non-editable, so `git pull` has no effect and the staleness detector reads
   mtimes of an installed snapshot. `nepal doctor` confirms.
2. **`nepal s02 --force`** (~7 min) — the act durations changed (Act 1 to 83 s,
   Act 5 to 311 s), so the music map is stale. Act 5 gets more music, Act 1
   less.
3. **`nepal s03 --redo shots,photos`** — re-detect with the 20 s shot cap,
   re-measure photographs on the corrected scale, then S03.3 metrics and S03.7
   gate run for the first time. Roughly an hour; proxies are untouched.
4. **Read the S03.3 percentile lines** it logs for sharpness, exposure,
   motion and stability, and calibrate against them: the `quality_curves`
   thresholds are the spec's, written before anyone had seen this corpus, and
   `process.metric_jerk_ref_px` is a guess. The spec expects roughly 70%
   rejection on camera material; check what the gate actually does.
5. **S03.4** (LUFS, wind, silero-VAD), then **S03.5** transcription — §1.4
   makes speech the spine of the film, so this is not optional. `has_speech`
   also switches on the gate's speech reprieve, which is inert until then.
6. **S03.6** faces, then M5 (S04 semantic + S05 scoring), then M7 (S06 assemble
   + S07 draft render).

## Open, and waiting on a person

- **Gate 1 FOV.** `fov_deg` solved to 193 as a low-confidence fallback, on a
  curve that was flat and non-convex because it was measuring seams on flat
  640×360 proxies that are not fisheye at all. Now that only genuine
  dual-fisheye frames reach the solver it is worth re-running (`nepal
  fov-check`) and confirming by eye.
- **`spine.whisper_language: ru`** is still marked "confirm".
- **AWS** is unauthorised in the session that built this. Nothing has been
  provisioned.

## Decided, do not relitigate

- Runtime is a **range**: `film.min_duration_s` 900, `target_duration_s` 1200,
  `max_duration_s` 2400. Twenty minutes is what it aims for, not a cap.
- Six creative changes were adopted (spec §1.2–§1.7): a cold open, altitude
  made visible, speech as the spine, exertion as the story, the runtime range,
  and a tonal constraint. Act 1 was halved; Act 5 grew and now covers
  Kathmandu, the Delhi layover and the flight home.
- Photographs are in the film, capped at `film.photo_share` (10%) **per act**,
  never ranked against footage.
- Glacier IR transition and `--proxy-only` ingest are both recommended
  **dropped** at this corpus size — see the README's deviations table.
