# Where the project stands

**As of 2026-09-16.** Keep this current: it is what a new session reads to
avoid re-deriving a fortnight of findings. When a stage lands or a number
changes, edit this file in the same commit.

## Build status

| Milestone | Scope | Status |
|---|---|---|
| 1 | S01 Probe + S02 Spine | **done**, validated end to end |
| 2 | S03 per-clip processing | **done** — S03.0–.7 all run on the corpus (.8 dropped) |
| 5 | S04 Semantic + S05 Score | S05 **done**; S04.1 built, not yet run; S04.2/.3 blocked on Bedrock |
| 7 | S06 Assemble + S07 Draft render | **done** — a first draft cut exists |
| 3, 4, 6, 8, 9 | containerise, Batch, gates, conform, Step Functions | not started |

S01–S03.4 ran locally. S03.5 ran on a rented EC2 box — see "The cloud move"
below. **That account is now suspended pending document verification**, so
everything since has run on this machine, and everything the film needs up to
Gate 3 has turned out to run here. Nothing of value is trapped in AWS: the
originals never left, and the DB, transcripts, face embeddings and proxies are
all local.

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
- **S03.2** (2026-09-15, `--redo shots`) — **1,152 shots** from 480
  recordings in 47 min (the 20 s cap divided the 395 cut-less recordings).
  Per act {1:111, 2:112, 3:335, 4:162, 5:422, None:10}; 774 positioned.
- **S03.0** (same run) — **704 photo shots** from 706 dated photos in 7:17
  (pillow-heif is core now, so HEIC counts). Per act {1:62, 2:131, 3:277,
  4:58, 5:176}; 2 outside every act.
- **S03.3** (same run) — 1,152 shots measured in 37:48. Percentiles
  (p5/p25/p50/p75/p95): sharpness 0 / 4.96 / 6.30 / 7.17 / 8.01 · exposure_pen
  0 / .001 / .006 / .11 / 1.0 · motion 0 / .12 / .53 / 1.47 / 4.51 · jerk (px at
  320 wide) camera p50 .24 p90 .83 p99 1.87, phones p50 .07 p95 .49. The p5 =
  0 / p95 = 1.0 tails are **five recordings that are black at source** (63
  min, camera in a bag), not a measurement problem.
- **S03.4** (2026-09-15, first run) — 1,151 shots in 3:13 (one kulikov
  recording has no audio track). audio_lufs p5/p50/p95 −58 / −27 / −17;
  wind share p50 0.36, p95 0.82, **65 shots flagged wind**; **speech 69.7 min
  in 555 shots — 32% of 219 min of footage**, against the spec's "typically
  under 10%". Camera 45 min (32%), kulikov phone 10, keller phone 9,
  telegram 5 (61%). Not wind heard as speech: speech share falls with wind
  share (48 → 26 → 16 → 0%) and the wind-flagged shots have none. The VAD is
  faster-whisper's bundled silero ONNX model, ~65× realtime on this CPU.
- **S03.5** (2026-09-15, on the EC2 box) — **537 shots, 66.2 min of speech,
  in 47:55** at 0.4× realtime, i.e. faster than the audio it is reading.
  45,235 characters of transcript; 2 speech shots came back empty. Locally
  the same work was measured at 27 h. Per source: camera 298 shots /
  27.5k chars, keller 86 / 6.4k, kulikov 132 / 7.5k, telegram 23 / 3.8k —
  telegram densest at 166 chars/shot, which is what a round video is.
  Spot-checked: coherent Russian, gear talk and reflection, not wind
  artefacts. `large-v3`, beam 5, `ru` forced.
- **S03.6** (2026-09-15, on the EC2 box) — **438 of 930 candidate shots show
  a face**, 1,168 faces, 15:41 at 1.0 s/shot **on CPU**: the G-instance quota
  never arrived and was never needed. Attribution after the merge pass:
  keller 256 shots, kulikov 92, 90 showing someone else. Cluster sizes
  482 / 264 / 88 / 20 / 18 / 12 and a long tail of ones.
  Two findings, both recorded in the code:
  - **The proxies were distorted.** 367 of 458 flat recordings are portrait
    (720x1280, or 1080x1920 behind a rotation flag) and were being squashed
    into a 16:9 box, stretched 2.4x and 3.2x. Every metric computed happily
    on them; it surfaced only when the detector returned nothing on a frame
    filled by a face. S07 renders the draft from these proxies, so the film
    would have had stretched people in it. Rebuilt, and 475 shots re-measured.
  - **The greedy clustering split each trekker across several clusters**
    (keller 376+65+25, kulikov 200+33) because a cluster mean drifts toward
    the poses it accumulates first. Merging cluster means at 0.40 fixed it;
    fragments sat 0.48-0.78 apart and strangers 0.00-0.12, so the threshold
    is not delicate.
- **S03.7** after calibration (below) — **1,626 of 1,856 survive** (12%
  rejected): soft 219 (214 black + 5), too short 7, shaky 4, exposure 0.
  Camera 429 of 433 real shots, phone 1,109 of 1,121, telegram 88 of 88.
  Survivors per act {1:173, 2:239, 3:578, 4:219, 5:408}. With S03.4's
  `has_speech` the reprieve is live: 1,632 survive (soft 217, short 6,
  shaky 1). The spec's "expect
  ~70% rejection on camera material" does not describe this corpus: the
  Insta360's single-lens mode and both phones stabilise in camera, so almost
  nothing is shaky, and the black recordings account for the rest.

### S05 + S06 + S07 — the first draft cut (2026-09-16)

`nepal cut` scores, assembles and renders in one driver.

- **S05** scored all 1,632 surviving shots. `score_sem` is **absent for every
  one of them** — captions are S04.3 and Bedrock is blocked — so `weighted`
  dropped the term and renormalised across tech and context. The ranking is
  therefore correct on what is known rather than distorted by a zero, but it
  has not been informed by what is *in* the shots.
- **Shortlist: 400** (199 video, 201 photo), of 1,632.
- **S06** laid out **220 slots, 19.8 min** — acts {1: 21 slots/1.4 min,
  2: 40/3.4, 3: 74/7.6, 4: 37/2.0, 5: 48/5.5}, 126 video and 94 photographs.
  `timeline.otio` and `timeline.fcpxml` are written beside the draft.
  **MMR ran without its diversity term** (no CLIP embeddings yet), so it
  degenerated to score order and the cut repeats itself in places.
- **S07** rendered `gates/gate3/draft.mp4` — **960×540, 203.5 MB, 8:18 wall
  clock**. Frames sampled across the runtime confirm real, varied material.

Three render defects were found by watching the output rather than by reading
ffprobe, and each is now covered by a test that runs the real binary:

- **`trim` after the decoder** produced *no output frames at all* in 2:46. A
  `trim` filter runs post-decode, so a shot 20 minutes into a recording costs
  20 minutes of decoding. Seeking moved to input `-ss`/`-t`.
- **94 photograph slots were silently dropped** — 8.7 min of a 19.8 min cut.
  The first "successful" draft was 9.9 min and looked fine. A slot with no
  media is now a loud warning.
- **`-loop` is private to the image2 demuxer** and fourteen of the stills are
  HEIC, which the ISO-BMFF demuxer reads: "Option loop not found". The slow
  test used `.jpg` and passed. The hold moved into the filter graph.

**No `drawtext`.** This ffmpeg is built without libfreetype, so the draft
carries no shot_id/timecode overlay and Gate 3 notes must cite wall-clock
times. The stage says so and renders anyway.

### The 1.1 minutes that were missing, and why (2026-09-16)

The render came out 18.7 min against a 19.8 min timeline. It was not ffmpeg
truncating arbitrarily. Measured against the recording durations: the 126
video slots asked for **664.9 s and only 593.3 s exists** — 71.6 s, which is
the whole discrepancy.

`assemble.lay_out` set each slot from the act's duration range scaled by the
shot's score and **never checked the shot's own length**. Two ways it bites:
a 1.87 s phone clip handed a 6.85 s slot in Act 5's 5–8 s regime, and a shot
starting 37.6 s into a 40.8 s recording asked for 6.9 s of the 3.1 s left.
The gate admits shots at `min_shot_s` **1.5 s** while acts ask for 2–8 s, so
the mismatch is structural, not a stray row.

Fixed: a slot is now `min(what the act wants, what the shot has)`, preferring
the latest beat that fits so the cut stays on the grid. A photograph is not
clamped — a still holds for as long as it is asked to.

**Consequence to decide at Gate 2/3:** the acts now come up ~1.2 min short of
their budget rather than claiming footage that does not exist. Refilling them
means selecting more shots, which is a shot-selection question and therefore
the operator's.

## Do these next, in order

1. ~~Editable reinstall~~ — **done 2026-09-15.** `nepal doctor` says
   `(editable -- this checkout is what runs)`. `nepal` is not on PATH unless
   `.venv` is activated; use `.venv/bin/nepal`.
2. ~~`nepal s02 --force`~~ — **done 2026-09-15**, see the run notes above.
3. ~~`nepal s03 --redo shots,photos`~~ — **done 2026-09-15**, 1:32 wall
   clock on this machine. Numbers above.
4. ~~Calibrate the gate~~ — **done 2026-09-15.** What the percentiles and the
   frames showed, and what moved (all in `config/pipeline.yaml` with the
   evidence in the comments):
   - `metric_jerk_ref_px` 4.0 → **1.2**. The guess was an order of magnitude
     high; with it, stability never fell below 0.84 and the rule was inert.
     Frame strips of the extremes: 3.5 px is a camera swinging at knee height
     pointed at boots; 1.9 px is a tilted, drifting, usable frame. The floor
     now cuts at ~2.2 px. The jerk here is not oscillation — the mean-flow
     series of the worst shots is smooth — it is the camera being swung.
   - `max_exposure_pen` 0.15/0.20/0.35 → **0.60/0.60/off**. At the spec's
     values the rule rejected 30 video shots and **22 of them were Keller's
     talking-to-camera diary selfies** with a white sky behind a correctly
     exposed face — the §1.4 spine — plus a teahouse at night and two round
     videos. Its 11 photo rejections were all Act 1 screenshots (the
     itinerary, the e-visa, the KTM–DEL flight). `exposure_pen` measures a
     scene's contrast range, not an exposure failure; genuinely blank frames
     sit at 1.0 and are caught by the sharpness floor anyway.
   - `shots.jerk_px` is now stored and **stability is re-derived at gate
     time**, so moving `jerk_ref` is a `nepal s03` re-run (seconds), not a
     38-minute re-measure. The 1,152 existing rows were backfilled by the
     exact inverse, checked against fresh OpenCV measurements to 4 decimals.
5. ~~S03.4~~ and ~~S03.5~~ — **both done 2026-09-15.** S03.4 stores
   `audio_lufs`, `wind_lf_share`, `speech_s`; `has_speech` and `wind` are
   derived at gate time from `process.speech_min_s` /
   `process.wind_lf_ratio`, so both thresholds move without a re-measure.
   S03.5 ran on the EC2 box in 48 minutes — see the run record above. The
   film now has its spine: 45k characters of Russian across 537 shots.
6. ~~S03.6 faces~~ — **done 2026-09-15**, see above. **Milestone 2 is
   complete.** Next is M5 (S04 semantic + S05 scoring), then M7 (S06 assemble
   + S07 draft render).
   - **S04.1 CLIP is the only remaining GPU candidate.** Do not wait on the
     quota case for it: S03.5 ran 32x faster than the local machine and
     S03.6 at 1.0 s/shot, both on CPU. Measure first.
   - **S04.3 captioning is Bedrock**, and eu-north-1 has every model worth
     using (Opus 5, Sonnet 5, Haiku 4.5, Fable 5.1, Nova) — **no region move
     needed.** Estimated from the real shot counts and the image-token
     formula: 1,632 shots × (4 frames @512px ≈ 196 tok each + text) ≈
     **1.63M input and 0.21M output tokens**, i.e. **$5.39 on Sonnet 5**,
     $2.69 on Haiku 4.5, $1.99 on Nova Pro — halved again by the Batch API,
     which suits a stage that is not latency-sensitive. The spec's $20–40
     assumed 2,500 shots at larger frames. `max_caption_shots` (2,500) does
     not bind. Re-measure against real `usage` once the form clears; these
     are published rates and Bedrock is partner-priced.
   - Gate 2 needs a representative face per cluster: the embeddings are in
     `work/faces/embeddings.npy`, and `nepal s03 --redo recluster` re-labels
     from them in seconds if a threshold moves.

7. ~~S05 + S06 + S07~~ — **done 2026-09-16**, see the run record above.
   **Milestone 7 is complete** and a draft cut exists.
8. **Gate 3 is open.** `work/gates/gate3/draft.mp4` is waiting on the
   operator. It is picture-only: no music bed, no mix, no overlay.
9. **S04.1 CLIP embeddings — run locally.** `torch 2.14.0+cpu` and
   `open_clip 3.3.0` are installed in `.venv`; measure before renting
   anything. This unlocks MMR's diversity term, which is the single biggest
   improvement available without Bedrock: the current cut repeats itself
   because MMR had nothing to compare shots with.
   - **Predicate fixed 2026-09-16.** S04.1 asked for `status = 'candidate'`,
     but S05 promotes its picks to `'shortlisted'` — so once a cut existed,
     the stage would embed 1,232 rows and skip the 400 the film is made of.
     It now asks for `status <> 'rejected'`, the complement of the gate's
     verdict, which cannot drift as statuses are added.
10. **Wire the music bed and the mix into S07.** The render is picture-only;
    `build_command` already takes `music_path` and normalises it, but S05's
    driver does not pass one, and the §7 ducking rules and music-out windows
    are not built.
11. **Add `-progress` to the ffmpeg invocation** so S07 reports a real
    percentage instead of the operator inferring one from file growth.

## The cloud move

**The AWS account is suspended** (2026-09-15) pending document verification;
the operator has submitted them. While it is down, Bedrock is unreachable, so
S04.2 framing, S04.3 captioning and S06's ordering refinement are all blocked.
Nothing else is: every remaining stage runs on this machine.

**Yandex Cloud is available as an alternative compute source** (operator,
2026-09-16). It is a real option for anything that is *just CPU or GPU* — it
has instances and S3-compatible object storage. It is **not** a substitute for
Bedrock: there is no Claude there, so captioning would need either the
Anthropic API directly (a spend decision, and the operator's) or an
open-weight VLM run on a rented GPU. Measure locally before renting either
way; that is how the G-instance quota turned out not to matter.

Done 2026-09-15, because S03.5 at 7–20 h locally was not worth waiting for.
**One rented box running the same `nepal` CLI, not the spec's Batch/ECR/Step
Functions architecture** — that was sized for 500 GB and 2,400 minutes; at
65 GB it is days of build for nothing the resumable CLI does not already do.
Recorded as a deviation in the README.

- **Account** eu-north-1, root session via `aws login` (the CLI session
  expires; re-run it when a call says so).
- **Budget** `NepalBudget` raised 40 → **60 USD**, alerts at 50% and 80% to
  the operator's address. Month-to-date spend was 0.
- **Bucket** `s3://nepalvideo-kk-eun1`, private, SSE-S3, public access
  blocked. Layout follows spec §2.2 (`raw/`, `work/`).
- **Instance** `m7i.2xlarge` **on-demand**, 8 vCPU / 30 GB, 250 GB gp3,
  `nepal-pipeline` SG (SSH from the operator's IP only), instance profile
  `nepal-pipeline-ec2` (that bucket only, plus SSM).
- **Not a GPU box.** The account's G/VT quota is **0 vCPU in every region
  checked** (eu-north-1/west-1/central-1/west-2/west-3, us-east-1/2,
  us-west-2), so `g4dn.xlarge` cannot launch at all. Increase requests for
  Spot and On-Demand G/VT are **CASE_OPENED**. Standard-family quota is 8
  vCPU, which is what `m7i.2xlarge` fits inside — and it is why the box is
  8 vCPU rather than 16.
- **On-demand, not Spot**, by the operator's call. Spot is ~4× cheaper
  ($0.103 vs $0.428/h measured) but the run is short enough that the
  interruption risk is not worth managing.
- **Measured: S03.5 ETA ~50 min on the box against ~27 h locally**, a ~32×
  speedup with no GPU at all — it is the RAM (30 GB, no swap) and 8 cores,
  not vector hardware. The local machine was swapping 2.6 GB.
- **Uploaded** `nepal_work` (proxies, audio, DB, transcripts), phones, chat,
  music — ~9 GB, the whole of what anything before S08 reads. The 60 GB of
  camera originals are syncing separately at low priority; only S08 conform
  needs them.
- **Scripts** `tools/cloud/launch.sh` and `tools/cloud/bootstrap.sh`. The
  bootstrap deliberately does not start a stage: the DB is synced last, after
  the local run has stopped.

**Where the database is authoritative moves with the work.** It went up at
the 2026-09-15 07:3x cutover, and came back down after S03.5 finished
(`nepal.sqlite.pre-cloud-*` next to it is the pre-sync local copy). Sync
before and after any remote stage, and check
`select count(*) from shots where transcript is not null` on both ends —
539 is the number as of now.

## Open, and waiting on a person

- **Gate 3: the draft cut.** `work/gates/gate3/draft.mp4`, 18.7 min. This is
  a creative judgement and it is the operator's. What it is missing, and why,
  is in the run record above — none of it is a defect to fix before watching.
- **AWS account suspended**, documents submitted, waiting. This blocks S04.2,
  S04.3 and S06 ordering refinement and nothing else.

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
- **Bedrock needs a one-time use-case form.** Every model in every region —
  Anthropic *and* Amazon Nova — returns `ValidationException: Operation not
  allowed` until it is submitted. The Model Access page is retired (models
  auto-enable on first invocation), so the form is the only gate left: Bedrock
  console → Model catalog → an Anthropic model. It is per account, shares
  details with Anthropic, and is the operator's to submit. Diagnosis note: the
  error is identical for a root session and an IAM role, so it is not the
  often-cited "Bedrock blocks root" — that hypothesis was tested and wrong.
  `bedrock:InvokeModel` is already on the `nepal-pipeline-ec2` role, which is
  the credential S04.3 should use.
- **G-instance quota: declined, and it does not matter.** AWS refused both
  requests (Spot `L-3819A6DF`, On-Demand `L-DB2E81BA`) as routine new-account
  ramping and offered an appeal. **Do not appeal.** Every stage that was
  supposed to need a GPU ran on the 8-core CPU box instead: S03.5 at 32x the
  local machine, S03.6 at 1.0 s/shot, S04.1 CLIP likewise. The spec's
  `g4dn.xlarge` was a reasonable guess that measurement retired.

## Known data notes

- **Five camera recordings are black at source** — `LRV_20240413_124619`,
  `…0424_013819`, `…0425_164006`, `…0425_170919`, `…0427_095106`, 63 minutes
  together, mean luma ~3/255 in the `.lrv` itself. The camera was recording
  inside a bag. Their 214 shots are rejected as soft; they cost 47 minutes of
  shot detection and metrics that a luma check at S01 would have saved, but
  the pipeline runs once and they are already processed.

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
