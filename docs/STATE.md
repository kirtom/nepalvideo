# Where the project stands

**As of 2026-09-15.** Keep this current: it is what a new session reads to
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

- **S02** (2026-09-15, `--force`, on the media machine) — 1:44 wall clock,
  of which 1:37 was S02.6 transcribing 12 round-video messages with
  `large-v3` on a 4-core, 11 GB box that was swapping; the "~7 min" estimate
  came from the cloud machine. S02.6 is resumable per file, so a killed run
  keeps its transcripts. Music: **23 tracks**, 3 stale rows removed
  (Alphaville, *Giorgio by Moroder*, Imogen Heap — files no longer in
  `music/`; the old map had Act 3 on a track that did not exist on disk).
  New map: Act 1 Underworld 83 s · Act 2 Murray Gold + Krishna Das 287 s ·
  Act 3 Daft Punk *The Grid* + Zack Hemsey + Rusted Root + Bill Withers
  407 s · Act 4 Edward Sharpe 113 s · Act 5 José González + Cinnamon Chasers
  311 s; Marusha held for credits; 10 of 23 tracks used; no map problems.
  Act boundaries unchanged. Transcripts are coherent Russian (gear talk in
  the planning phase), which is evidence for `whisper_language: ru` though
  the language is forced, not detected.
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

1. ~~Editable reinstall~~ — **done 2026-09-15.** `nepal doctor` says
   `(editable -- this checkout is what runs)`. `nepal` is not on PATH unless
   `.venv` is activated; use `.venv/bin/nepal`.
2. ~~`nepal s02 --force`~~ — **done 2026-09-15**, see the run notes above.
3. **`nepal s03 --redo shots,photos`** — **running as of 2026-09-15 04:12.** Re-detects with the 20 s shot cap,
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

- **Gate 1 FOV.** `fov_deg` = 193 in `decisions` is a **carried-over
  fallback**: `work/reports/s01_probe.json` shows the last S01 ran with
  `--skip-fov`, so the value predates frame-shape classification and was
  measured on flat 640×360 proxies that are not fisheye at all. The solver
  has not run since only genuine dual-fisheye frames reach it. Re-solving
  needs `nepal s01 --force` (there is no FOV-only redo) and then `nepal
  fov-check` to confirm by eye. Note S03.1 proxies for the 22 dual-fisheye
  recordings were reprojected at 193; a materially different FOV would mean
  rebuilding those 22.
- **`spine.whisper_language: ru`** is still marked "confirm".
- **AWS** is unauthorised in the session that built this. Nothing has been
  provisioned.

## Known data notes

- **Three `phone_kulikov` clips are stamped 2025-11-22** (`video_1_902aaa…`,
  `4F9001FC…`, `5822EE80…`; 720×1280, 9–32 s). Every embedded time tag agrees
  on that date and none carries `Keys:CreationDate`, so it is an export date
  and the capture time is unrecoverable from metadata. S02 correctly refuses
  to stretch Act 5 to them; they sit outside every act. If the capture dates
  are known, a manual override would place them.
- **SRTM tiles N28E077, N55E037, N59E030 are absent by design.** They lie
  under Delhi, Moscow and St Petersburg; `nepal fetch-reference` covers the
  trek extent only and ignores fixes far from the route. 45 home/transit
  photos have null altitude as a result, which the film does not need.
- **This machine:** 4 cores, 11 GB RAM, heavy swap under `large-v3`. Expect
  S03.5 transcription to be bounded by memory, not CPU. Closing browsers
  before a whisper run roughly triples its speed.

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
