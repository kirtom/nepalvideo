# Nepal Trek Film Pipeline

Turns several hundred gigabytes of raw trek media — a 360 action camera, two
phones, a Telegram export and a music library — into a ~20-minute documentary
plus a 3-minute short cut, with three human approval gates.

`docs/spec.md` is the specification this is built to. `docs/STATE.md` is where
the project actually stands right now, with the numbers from the last real run
and the current issue list. `docs/DATA_AND_PIPELINE.md` catalogues every kind
of raw material and every transformation applied to it. `CLAUDE.md` is how to
work in this repository. This file is the tour.

---

## What it does

The pipeline reads media that stays where it is, and writes everything it
derives into a work directory beside it. Nothing is edited by hand until the
first approval gate.

| Stage | What it works out |
|---|---|
| **S01 Probe** | Every file's real identity: what it is, when it was shot, which chapters belong to one recording, the lens FOV, and how far each device's clock is wrong |
| **S02 Spine** | The story: a merged GPS track, altitude, place names, the Telegram conversation, act boundaries, and which music goes where |
| **S03 Process** | Per clip: proxies, shot boundaries, technical metrics, audio, speech, and a quality gate |
| **S04 Semantic** | Embeddings, 360 framing decisions, captions *(not built)* |
| **S05 Score** | Ranking and a shortlist *(not built)* |
| **S06–S09** | Assembly, draft render, conform, delivery *(not built)* |

### The film it is aiming at

Five acts, a **runtime range** rather than a fixed length (15–40 min, aiming at
20), and six creative constraints from the brief: a cold open carved off the
top, altitude made visible, **speech as the spine**, exertion as the story, and
a tonal constraint. Photographs are in the film but capped at 10% of each act,
never ranked against footage — a still wins every stability contest by not
moving. Act 4 is one swell into a hard cut to silence.

---

## Status

| Milestone | Scope | Status |
|---|---|---|
| 1 | S01 Probe + S02 Spine | **done**, validated end to end on the real corpus |
| 2 | S03 per-clip processing | **done** — S03.0–.7 all run on the corpus (.8 dropped) |
| 5 | S04 Semantic + S05 Score | S05 **done**; S04.1 built, not yet run; S04.2/.3 need Claude |
| 7 | S06 Assemble + S07 Draft render | **done** — a draft cut exists |
| 3, 4, 6, 8, 9 | containerise, Batch, gates, conform, Step Functions | not started / partly dropped |

**A complete film exists end to end**: 1,364 raw assets → 1,856 shots → 1,632
past the gate → 400 shortlisted → 220 slots → an 18.7-minute draft.

Everything runs locally today. S03.5 transcription was the one stage that
justified renting a machine (7–20 h locally against 48 min on 8 cloud cores);
it has since run. The remaining cloud dependency is **Claude, for S04.2
framing, S04.3 captioning and S06 ordering** — which is what the draft is
missing. See [Running it in the cloud](#running-it-in-the-cloud) and the
current issue list in `docs/STATE.md`.

---

## Install

```bash
apt-get install -y ffmpeg libimage-exiftool-perl     # or: brew install ffmpeg exiftool

python3 -m venv .venv
.venv/bin/pip install -e '.[vision,music,asr]'
```

**Install editable.** A non-editable install means `git pull` changes nothing
about what runs, and the staleness detector reads the mtimes of an installed
snapshot rather than your working tree. `nepal doctor` says plainly which one
you have.

Extras, and what each buys:

| Extra | For | Without it |
|---|---|---|
| *(core)* | S01, S02 spine, photo stills | — |
| `vision` | S03.2 shot detection, S03.3 metrics | no shots, no metrics |
| `music` | S02.7 music analysis | no music map, so no act scoring |
| `asr` | S02.6 and S03.5 transcription | speech — the film's spine — stays unwritten |
| `semantic` | S04 embeddings (torch, CUDA) | S04 cannot run |
| `cloud` | boto3, for the S3 sync on the rented box | local only |

`pillow-heif` is **core, not an extra**: iPhones shoot HEIC by default and
Pillow cannot decode it alone. Without it S03.0 silently loses roughly half the
photographs — 339 of 706 on this corpus.

Check the install before anything heavy:

```bash
.venv/bin/nepal doctor
```

It prints the interpreter, whether the checkout is what actually runs, every
resolved path, which external binaries are present, and which optional packages
are missing along with the extra that provides each.

> `nepal` is only on your `PATH` if the venv is activated. `.venv/bin/nepal`
> always works and is unambiguous about which interpreter it uses.

---

## Layout

Three directories, deliberately separate:

```
nepalvideo/          this repository — code, config, docs
nepal_data/          the delivered media (~65 GB), read-only in practice
nepal_work/          everything the pipeline derives (~4 GB), disposable
```

`work_root` sits beside the media rather than inside the repo because S03
writes a proxy, an audio track and sampled frames per recording. Relative paths
in the config resolve against the **project root** (the folder holding
`config/`), not your shell's working directory, so `nepal` behaves the same
wherever you invoke it from.

```yaml
project:
  data_root: /data/projects/nepal_data
  work_root: /data/projects/nepal_work
```

### What lands in `nepal_work/`

| Path | Written by | Contents |
|---|---|---|
| `db/nepal.sqlite` | all stages | the single source of state (spec §5) |
| `reports/s0*.json` | each stage | what that run actually did, for the next session |
| `proxies/*_eq.mp4` | S03.1 | equirect proxies, **video only** |
| `audio/*.wav` | S03.1 | 16 kHz mono PCM — the clock solver, VAD and ASR all read these |
| `views/`, `frames/` | S03.1, S04 | rectilinear yaw renders and sampled stills |
| `transcripts/` | S02.6, S03.5 | one JSON per round video and per speech shot |
| `music/music_map.json` | S02.8 | which track plays over which act, with beat grids |
| `gpx/trip.gpx` | S02.1 | the merged track |
| `fov/`, `clock/` | S01.4, S01.5 | solver working files and Gate 1 comparison sheets |

`data/` inside the repo holds fetched reference data (SRTM tiles, GeoNames) and
is git-ignored. Recreate it any time with `nepal fetch-reference`.

---

## Running it

```bash
nepal doctor                        # check the install and the paths
nepal s01                           # manifest, chapters, FOV, clock offsets
nepal fetch-reference               # SRTM tiles + GeoNames gazetteer (once)
nepal s02                           # track, altitude, places, telegram, music
nepal report                        # the chronological table — the checkpoint
nepal decisions                     # auto-solved values, with confidence
nepal s03                           # per-clip processing
nepal diagnose                      # when the checkpoint table looks wrong
nepal fov-check                     # Gate 1 seam comparison sheets
nepal prune [--dry-run]             # remove what the pipeline no longer produces
```

**Every stage is resumable.** Each sub-step records completion in
`stage_units`, so a re-run redoes only what is missing, and several sub-steps
are resumable through the data itself — a shot with a `sharpness` has been
measured, so a run killed halfway keeps everything it finished.

`--force` recomputes everything. For S03 that means rebuilding every proxy:
three and a half hours. Prefer naming the sub-steps:

```bash
nepal s03 --redo shots,photos       # re-detect and re-measure; proxies untouched
nepal s03                           # the gate always re-runs; everything else resumes
```

Valid `--redo` steps: `proxies,shots,photos,place,metrics,audio,asr,faces,recluster,gate`.
S01 and S02 take `--redo` too (`manifest,chapters,fov,clock` and
`gps_track,telegram,geotag,asr,music,acts`), so re-solving the FOV or
re-reading the Strava export no longer means `--force` and an hour of clock
correlation.

**`db.upsert` never removes**, so `nepal prune` exists: it re-derives the
recording grouping from the assets already in the database and deletes the
recordings, shots, timeline rows and work files that the current rules no
longer produce. `--dry-run` reports what would go. Run it after any change to
the grouping rules or to `nepal_data/`.

A re-run without `--force` warns when a completed unit was produced by an older
version of the code, so a report that changed nothing cannot be mistaken for
one that did work.

### How long each stage takes

Measured on the real corpus (480 recordings, 4.9 hours of footage, 706 photos)
on a 4-core, 11 GB machine:

| Stage | Wall clock | Note |
|---|---|---|
| S01 | tens of minutes | exiftool over ~1,250 files, plus the FOV and clock solvers |
| S02 | 1:44 | of which 1:37 was transcribing 12 round videos with `large-v3` |
| S03.1 proxies | 3:38:00 | the expensive one; `v360` reprojection dominates |
| S03.2 shots | 47 min | PySceneDetect over every proxy |
| S03.0 photos | 7 min | decode-bound, scales nearly linearly with cores |
| S03.3 metrics | 38 min | 5 sample points × 3 frames per shot |
| S03.4 audio | 3 min | loudness per shot, VAD once per recording |
| S03.5 speech | **48 min on EC2** (est. 7–20 h locally) | 537 shots, 66 min of speech, 0.4× realtime |
| S03.6 faces | 16 min on EC2 | 930 shots at 1.0 s/shot, CPU; no GPU needed |
| S03.7 gate | seconds | pure function of stored metrics and thresholds |
| S04.1 CLIP | **~6 h measured, not yet run** | ViT-L-14 at 13.1 s/shot on 4 cores; frame decode is only 0.64 s/shot |
| S05 score | seconds | pure function of stored metrics |
| S06 assemble | seconds | MMR plus the beat grid, both pure |
| S07 draft render | 8:18 | 220 slots to 18.7 min of 960×540 |

---

## The stages in detail

### S01 — Probe

Establishes what each file *is*, rather than trusting its name.

| Decision | How | Fallback |
|---|---|---|
| `frame_shape` | Classified on the actual frame geometry | — |
| `fov_deg` | Sweeps 188–206° and picks the FOV leaving least discontinuity at the two hemisphere meridians | 193°, flagged for Gate 1 |
| chapter grouping | Filename convention plus continuity of capture time | one file, one recording |
| `clock_offset_*_s` | Coincidence voting proposes candidates, GCC-PHAT audio correlation arbitrates | manual entry at Gate 1, prefilled |
| `clock_reference` | Whichever device's timestamps agree most closely with satellite time in `GPSDateTime` | keller |
| `camera_has_gps` | Direct probe | — |

**Clock offsets are the highest-consequence values in the project.** A wrong
one silently misaligns every downstream join — geotagging, message anchoring,
act assignment — and produces a plausible-looking but wrong film. They are
surfaced at Gate 1 regardless of confidence.

Solving them is two-stage because one stage cannot span the errors that occur
in practice: a flat battery resets an action camera's clock, and the resulting
error is *days* wide — thousands of times the ±600 s window audio correlation
searches. On this corpus the camera was about fourteen days out. Accuracy
ceiling is ±0.5 s, because device timestamps are whole-second; that is
comfortably inside what geotagging and act assignment need.

### S02 — Spine

Builds the story the film will follow.

- **S02.1 GPS track** — merges, in order of trust, the Strava export in
  `nepal_data/strava/` (an Apple Watch: a fix a second, barometric altitude
  and heart rate, one FIT file per trekking day), any shared GPX, and phone
  photo EXIF; drops points implying travel above 60 m/s. The activities
  become one row per trekking day with the day's name and its real start.
- **S02.2–.4 Place everything** — interpolates position for assets without
  their own fix, attaches SRTM altitude and a GeoNames place name. Refuses to
  interpolate across gaps longer than `spine.max_interp_gap_s`.
- **S02.5 Telegram** — parses the export into messages, classifies planning vs.
  after against the trek envelope, extracts a vocabulary of names, places and
  recurring jokes (S04 injects this into caption prompts, without which
  captions come back in generic model English).
- **S02.6 Round videos** — transcribes Telegram round video messages. The best
  narration material in the project: face plus voice, timestamped and
  unperformed.
- **S02.7 Music** — tempo, key, energy, dynamic range and beat grid per track.
- **S02.8 Music map** — Hungarian assignment of tracks to acts on normalised
  features, with a callback bonus for Act 5 sharing artist or key with Act 1,
  and one track held back for the credits.

Act boundaries come from a change-point fit on the altitude-vs-day curve, with
Act 1 taken from the message-phase instead (planning happens before the
altitude curve exists).

### S03 — Process

- **S03.0 Photographs** — each dated photo becomes a shot held for
  `photo_slot_s`, measured on the *same* sharpness and exposure scale as video.
  They were once on a different scale, which let the photo gate pass everything
  and saturated `score_tech`.
- **`place`** — every shot, video or still, gets a position interpolated from
  the track, an altitude (barometric between watch fixes, DEM otherwise), a
  place name and a trek day. Always re-run, because the track and the act
  boundaries move whenever S02 does; a photograph keeps its own fix and gains
  only what the fix lacks. Before this existed, video shots had none of it
  and the context score favoured stills.
- **S03.1 Proxies** — one decode, multiple encodes: a 1024×512 equirect proxy
  and a 16 kHz mono audio track per recording. Audio is written to its own
  `.wav`, not muxed into the proxy. The four rectilinear yaw views the spec
  asks for are deferred: S05 needs one yaw per *surviving shot*, not four per
  recording, so they are rendered as stills later. Phase A measured 3.46×
  realtime against 0.73× for the single-pass design.
- **S03.2 Shots** — PySceneDetect `ContentDetector`. A 360 camera on a walking
  person does not cut: 395 of 480 recordings came back with no boundary at all,
  one of them 29 minutes long. Anything longer than `process.max_shot_s` (20 s)
  is divided into equal pieces, because a shot is what the timeline chooses
  between and neither a metric nor a cut survives that length.
- **S03.3 Metrics** — sharpness, exposure penalty, motion magnitude and
  stability, sampled at five points per shot, three consecutive frames each.
  Three refinements forced by this corpus: three frames not one (motion between
  frames seconds apart is displacement, not motion); **jerk, not variance** —
  shake is the second derivative, since a steady pan has a large constant flow
  field and a shaken camera has one that reverses; and sharpness and exposure
  measured on the **central band** of an equirect frame, because the poles
  stretch sky and boots across the full width.
- **S03.4 Audio** — EBU R128 integrated loudness per shot, the share of
  spectral energy below 200 Hz (wind is broadband rumble that lives there), and
  silero-VAD speech. The VAD runs **once per recording**, not per shot: a
  sentence does not care where the shot detector put a boundary.
- **S03.5 Transcription** — faster-whisper over only the shots carrying speech,
  on the shot's own window, with segment times put back on the recording's
  clock. Writes each transcript as it goes, so a killed run keeps its work.
- **S03.6 Faces** — InsightFace `buffalo_l` over rectilinear views sampled
  out of the equirect frame at several yaws, 512-d ArcFace embeddings, then
  one greedy agglomerative pass plus an average-linkage merge over the whole
  corpus. 1,168 faces, 202 clusters. The embeddings are stored, so moving a
  threshold re-labels in seconds instead of re-detecting for hours.
- **S03.7 Quality gate** — see below.

### S04 — Semantic

- **S04.1 CLIP** — one embedding of the sharpest sampled frame of every
  surviving shot, through ViT-L-14, stored as a plain `.npy` with a JSON
  index rather than database rows: the spec asks for this artefact to outlive
  the film as a searchable index of a personal video library. Built and
  measured (~6 h on 4 CPU cores); not yet run to completion.
- **S04.2 Framing / S04.3 Captions** — *not built.* Both need Claude.

### S05 — Score

Three scores combined: `score_tech` (sharpness, exposure, stability, duration
fit), `score_sem` (VLM interest, CLIP–act similarity) and `score_ctx` (faces,
speech, a new maximum altitude, first-at-place, proximity to a chat message).
**A term with no data is dropped and the rest renormalised**, never counted as
zero — which is why the ranking survives having no captions.

### S06 — Assemble

Greedy MMR — `argmax(score − λ · max similarity to what is already chosen)` —
under hard constraints, laid onto the music's beat grid. Slot duration is
`min(what the act's range asks for, what the shot actually has)`. Exports
OpenTimelineIO and FCPXML 1.9, both written by hand with times quantised to
frame boundaries, so the cut is refinable in Resolve or Final Cut rather than
only re-runnable.

### S07 — Draft render

One ffmpeg invocation, conformed **from the proxies, never the originals** —
re-reading 5.7K H.265 for a 960×540 preview would cost hours to look identical
at that size. Video seeks at the input; photographs are held in the filter
graph. The mix (`process/mix.py`) is written but **not yet wired in**, so the
draft is currently picture-only.

---

## The quality gate

Rejects what is not worth a caption, a transcription or a human's attention,
judged against the curve appropriate to each asset's source:

```yaml
quality_curves:
  camera:   {min_sharpness: 4.0, max_exposure_pen: 0.60, min_stability: 0.35, min_duration_s: 1.5}
  phone:    {min_sharpness: 3.5, max_exposure_pen: 0.60, min_stability: 0.30, min_duration_s: 1.5}
  telegram: {min_sharpness: 2.0, max_exposure_pen: 1.00, min_stability: 0.15, min_duration_s: 1.0}
```

Three properties worth knowing:

**A shot carrying speech is judged on a floor of its own.** The brief makes
voice the spine of the film: a soft, wobbly frame under a sentence that carries
the story is worth more than a sharp frame of nothing, and the picture can be
cut away from while the audio keeps running. Where `has_speech = 1` the
sharpness floor is multiplied by `gate.speech_sharpness_factor`, the minimum
duration drops, and the stability floor does not apply at all. Exposure still
does — a blown-out frame carries no picture at any length.

**A metric that was never measured does not reject.** A photograph has no
stability by design; a shot whose proxy failed has nothing at all. The gate
removes the demonstrably unusable, not the unknown.

**The gate is re-run on every invocation** rather than skipped as done. It is a
pure function of the metrics and the thresholds, both of which move while the
film is being tuned, and a rejection left over from an older threshold is
invisible — the shot simply never appears again. Rejected shots stay in the
table as rows, so a threshold change brings them back without re-decoding
anything.

### Thresholds are stored as measurements, not verdicts

`shots.jerk_px`, `speech_s` and `wind_lf_share` hold what was *measured*;
`stability`, `has_speech` and `wind` are derived from them at gate time against
the current thresholds. So moving `process.metric_jerk_ref_px` or
`process.speech_min_s` costs a `nepal s03` re-run of seconds, not a 38-minute
re-measure of the corpus.

### What calibration against the real corpus changed

The spec's thresholds were written before anyone had seen this footage. Both
were wrong in the same direction — rejecting the film's best material:

- `metric_jerk_ref_px` **4.0 → 1.2**. An order of magnitude high: stability
  never fell below 0.84, so the rule was inert. Camera shots measure p50 0.24
  px, p99 1.87. Frame strips of the extremes showed 3.5 px is a camera swinging
  at knee height pointed at boots, and 1.9 px is a tilted but usable frame.
- `max_exposure_pen` **0.15/0.20/0.35 → 0.60/0.60/off**. At the spec's values
  the rule rejected 30 video shots, and **22 were talking-to-camera diary
  selfies** — a correctly exposed face against a white sky, which is exactly
  the §1.4 spine. Its photo rejections were all Act 1 screenshots. The measure
  describes a scene's contrast range, not an exposure failure.

The spec's "expect roughly 70% rejection on camera material" does not describe
this corpus. The Insta360's single-lens mode and both phones stabilise in
camera, so almost nothing is shaky; actual rejection is 12%, and most of it is
five recordings that are **black at source** — the camera recording inside a
bag for 63 minutes.

---

## The database

One SQLite file, a few thousand rows, at `work/db/nepal.sqlite`. No vector
database is warranted at this scale.

| Table | Holds |
|---|---|
| `assets` | one row per file: identity, capture time, position, frame shape, quality curve |
| `recordings` | chapters grouped into one take |
| `shots` | the unit of selection — metrics, audio, speech, captions, scores, status |
| `gps_points` | the merged track |
| `messages` | the parsed Telegram conversation |
| `music_tracks`, `music_sections`, `beats` | the music analysis and its grid |
| `timeline` | the cut, once S06 builds it |
| `decisions` | every auto-solved value with a confidence and a method |
| `stage_units` | the resumability ledger |

`decisions` is worth reading directly (`nepal decisions`): it is the audit trail
for every value a human might otherwise have to guess.

> **`db.upsert` never removes.** Anything a rebuild no longer produces —
> assets, recordings, shots — survives unless it is explicitly deleted. Shot
> re-detection deletes a recording's shots before inserting for exactly this
> reason.

---

## Configuration

**Every tunable lives in `config/pipeline.yaml`**, reachable by dotted path, so
call sites read like the spec section they implement. No magic numbers in code.
Duplicate keys raise rather than silently shadowing — a second `process:` block
once deleted the first and cost a 3.5-hour run.

The sections, roughly in pipeline order:

| Section | Governs |
|---|---|
| `project` | data root, work root, database path |
| `film` | runtime range, act allocation, photo share, cold open, selectivity |
| `acts` | per-act target/min/max seconds |
| `probe` | FOV sweep, clock solver windows |
| `spine` | interpolation gaps, geocoding, whisper model and language |
| `process` | proxy size and rate, shot detection, metric sampling, audio, ASR |
| `quality_curves` | the S03.7 gate, per source class |
| `gate` | the speech reprieve |
| `semantic`, `score`, `assemble`, `render`, `music` | S04–S07 *(not yet built)* |

The comments in that file are the interesting part: most values have a
paragraph explaining what was measured to arrive at them, and several record
what the wrong value did.

---

## The three approval gates

These are creative judgements and are never made by the pipeline. It surfaces
what is needed to decide, and stops.

1. **Gate 1 — FOV and clocks.** `nepal fov-check` renders seam comparison
   sheets at each candidate FOV alongside the measured discontinuity curve.
   Confirm by eye; confirm the clock offsets regardless of their confidence.
2. **Gate 2 — Shot veto.** Review the shortlist, veto shots, confirm the face
   cluster labels.
3. **Gate 3 — Draft cut.** Approve the rough cut before conform.

Milestone 6 wires these into Step Functions with `waitForTaskToken`. Until
then they are manual checkpoints.

### Manual checkpoints before that

1. **After `nepal s01`** — read `nepal decisions`. Is `fov_deg` plausible
   (188–206, and not the 193 fallback unless the material is poor)? Do the
   clock offsets look like real drift rather than noise? Check
   `work/reports/s01_probe.json` for chapter-continuity warnings and collapsed
   duplicates.
2. **After `nepal s02`** — run `nepal report`. *This is the checkpoint that
   decides whether the project is worth continuing.* Does the `pre` row show
   enough planning material for Act 1? Does altitude peak on the day you
   remember? Do act boundaries land where the trek changed character? Does the
   music read right, and does Act 5 echo Act 1?
3. **Review music exclusions** in `work/reports/s02_spine.json`. Cyrillic
   detection is exact; the romanised-artist list is best-effort. Correct it
   either way with `music.keep_artists` / `music.exclude_artists`.

---

## Running it in the cloud

**Nothing computes on the operator's machine.** Every stage, test, probe and
render runs on one GCP Spot VM, driven from here:

```bash
nepal remote status                    # is the box up, what has the project spent
nepal remote up                        # create or start nepal-cpu, wait for READY
nepal remote run s03 --redo place      # a stage on the box, log streamed here
nepal remote exec -- .venv/bin/python -m pytest -q      # anything else on the box
nepal remote pull                      # bucket -> local: db, reports, gates, ...
nepal remote down                      # stop; the disk and its contents stay
nepal remote up --gpu                  # the L4 profile, for CLIP and upscaling
```

**The bucket is the hub.** `gs://nepalvideo-29922345852` (europe-west4) holds
`raw/` (phones, chat export, music, Strava — not the camera originals until
conform), `work/` (proxies, audio, transcripts, faces, the database, reports)
and `ref/` (SRTM tiles, GeoNames). `nepal remote push` sends the working set
from here; the box pulls `raw/` and `work/` before every run and pushes
`work/` back afterwards, whatever the exit code, so a killed run keeps its
checkpoints; `nepal remote pull` brings the parts a run can change back here
to look at. From the first push on, the database in the bucket is the
authoritative one.

**Two profiles**, both Spot, both stopped rather than deleted on `down`:

| Profile | Machine | For | Spot price (estimate) |
|---|---|---|---|
| `cpu` (default) | `e2-standard-8`, 60 GB | everything that is CPU, and the API loops | ~$0.11/h |
| `gpu` | `g2-standard-4` + 1× L4, 100 GB | CLIP embeddings, faces re-runs, upscaling | ~$0.30/h |

Spot boxes stop on preemption (`--instance-termination-action=STOP`), so a
preempted run is resumed by `nepal remote up` and the resumable CLI. The box
gets the `storage-rw` scope and nothing else.

**What the box runs on first boot** (`tools/cloud/bootstrap-gcp.sh`, a GCE
startup script, so it runs on *every* boot and every step is idempotent):
ffmpeg, exiftool, git, a venv with every extra; a clone of the branch named
in `cloud.gcp.branch`, reset to `origin/<branch>` on each boot and each run
— **push before `nepal remote run`**; the bucket pulled to the same paths the
config names (`/data/projects/nepal_data`, `/data/projects/nepal_work`), so
`pipeline.yaml` needs no change; `/data/projects/READY` when done, which is
what `up` waits for. On the `gpu` profile it also installs the NVIDIA driver
with Google's installer if `nvidia-smi` is absent.

**Credentials.** The operator's `gcloud` login drives everything from here;
nothing is stored in the repo. The Anthropic key reaches the box as the
instance metadata attribute `anthropic-api-key`, which the startup script
exports into the `nepal` user's profile; set it with
`gcloud compute instances add-metadata nepal-cpu --metadata anthropic-api-key=…`.

**The ledger.** `work/reports/spend.json` records every VM hour (written by
`down`, at the profile's estimated price) and every paid API call (written by
the stage that made it, from the response's `usage`). `up` and every paid
stage ask it first and refuse past `cloud.spend_ceiling_usd` (15). The GCP
billing budget ("nepal", 60 USD, alerts at 50% and 80%) is the warning
behind the refusal.

**A host that holds part of the corpus.** The box never sees the 60 GB of
camera originals, so the manifest leaves the rows of any source whose
directory is absent on the current host untouched instead of treating them
as deleted files, and a file whose size and mtime match its row is not hashed
again.

### What was learned on AWS before the account was suspended

One rented box running the same CLI, not the spec's Batch + ECR + Step
Functions design: sized for 500 GB and 2,400 minutes, at 65 GB it was days
of build for nothing the resumable CLI does not already do. **No GPU was
needed for anything before CLIP**: S03.5 transcription dropped from ~27 hours
locally to ~50 minutes on eight cores and 30 GB — the win was RAM, the local
machine was swapping under `large-v3` — and S03.6 faces ran at a second a
shot on the same CPU. The AWS scripts (`tools/cloud/launch.sh`,
`bootstrap.sh`) are kept for the record; the account is suspended.

---

## Testing

```bash
pytest                   # fast unit tests, no media (~720 tests, ~80 s)
pytest -m slow           # end to end against ffmpeg-generated fixtures (~2 min)
python tools/make_fixtures.py /tmp/fixtures --quick
```

The fixture generator plants known ground truth — a dual-fisheye projection at
a known lens FOV, phone clocks skewed by known amounts, chapter-split takes — so
the tests assert *recovery of the truth*, not merely that the code runs.

**Verify at the real boundary.** Every metric this pipeline has got wrong was
wrong where Python meets ffmpeg, exiftool or OpenCV, and passed its unit test
because the test fed it a hand-built dict the real scan never produces. When a
change touches what a tool is asked for or what comes back, add a slow test
that runs the actual tool.

Two tests take a path to real material through the environment, because no
synthesiser is available to fabricate speech:

```bash
NEPAL_SPEECH_WAV=/path/to/16k-mono-speech.wav pytest -m slow
```

---

## Code layout

```
config/pipeline.yaml       every tunable in the spec, in one place
docs/spec.md               the specification
docs/STATE.md              where the project stands, with real numbers
src/nepal/
  cli.py                   the one entry point
  config.py                dotted-path config access
  db.py                    SQLite schema (spec §5), migrations, resumability
  freshness.py             "this was produced by older code" detection
  select.py                photo/video mix rules shared by S05 and S06
  diagnose.py              spine diagnostics
  reference.py             SRTM tiles and the GeoNames gazetteer
  util/                    ffmpeg/exiftool wrappers, hashing, progress
  probe/                   S01: manifest, chapters, FOV solver, clock solver
  spine/                   S02: gps, dem, geocode, telegram, music, acts, effort
  process/                 S03: reproject, shots, stills, metrics, audio, asr, gate, mix
  stages/                  the three stage drivers
tools/
  make_fixtures.py         synthetic nepal_data/ with ground truth
  fetch_reference.py       SRTM + GeoNames
  survey_data.sh           read-only survey of a delivered nepal_data/
  survey_camera.sh         camera-clock and music deep dive
  cloud/launch.sh          launch the rented box
  cloud/bootstrap.sh       what that box runs on first boot
tests/                     unit tests + end-to-end
```

**Design rule:** modules that shell out keep the shelling in a thin wrapper, and
the decision logic that consumes the output stays pure. That is why the FOV
solver, the clock solver, the chapter grouper, the gate and the audio measures
are all unit-testable with no media present.

---

## Traps this project has already fallen into

Each of these cost a wrong diagnosis or a wasted multi-hour run.

- **Extension does not tell you what a file is.** `.insv` holds dual-fisheye,
  single-lens and flat 16:9 alike. Classify on frame shape. Fixing this took
  chapter-continuity warnings from 358 to 15.
- **A tag that is ranked but never requested does not exist.** `exiftool` is
  called with an explicit tag list; a reader can rank `CreationDate` as highly
  as it likes and never see it.
- **`Keys:CreationDate` is the only capture stamp that survives export, AirDrop
  and iCloud**, and the only one carrying its own UTC offset.
- **Instrument before theorising.** Four wrong diagnoses of the S03.1 slowdown
  were reasoned from code; per-recording logging plus proxy mtimes answered it
  in one round.
- **`db.upsert` never removes.** See [The database](#the-database).
- **A threshold written before the data existed is a guess.** Both gate
  thresholds rejected the film's best material until they were measured against
  the corpus.
- **Do not optimise speculatively.** This pipeline runs once. A measured 3×
  is worth taking; a plausible one is not.

---

## Deviations from the specification

Each is a deliberate change with a reason, not an oversight.

| Spec says | Built instead | Why |
|---|---|---|
| Bundle an offline Nominatim extract for reverse geocoding | GeoNames country dump plus a KD-tree | Nominatim needs PostgreSQL, PostGIS and a multi-gigabyte OSM import to name about fifty cluster centroids. A GeoNames dump is a few megabytes of text with no service to run, and it returns villages and peaks rather than postal addresses — which is the question a trek actually asks. |
| Read SRTM tiles (implying a geo stack) | `numpy` directly on `.hgt` | An `.hgt` file is raw big-endian int16 with no header. GDAL and rasterio are a large dependency for one array lookup. Validated against real NASA tiles: Tengboche +2 m, Lukla −10 m, Dingboche −45 m. |
| Photos are the geolocation spine | Strava first, photos fill the gaps | The operator's watch tracked every trekking day at a fix a second with barometric altitude and heart rate. Photo EXIF still covers the jeep days and the cities, where the watch did not run. |
| Camera is the reference clock | The GPS-validated phone is | Photos with a GPS fix also carry satellite time, so `DateTimeOriginal` against `GPSDateTime` is direct evidence of which clock to trust. On this corpus the camera is provably the wrong one. |
| Clock offsets from audio cross-correlation over ±600 s | Coincidence voting proposes candidates, then audio arbitrates | The delivered camera's clock is about fourteen days out — roughly two thousand times the ±600 s search window, so the audio stage alone would never have found it. |
| Music features from audio files | Also from a playlist export | The delivered `music/` held a playlist CSV and no audio. Spotify audio features cover four of the five dimensions the act assignment needs; the fifth, dynamic range, has no playlist analogue and is masked out of the distance rather than approximated. |
| Four rectilinear yaw videos per recording in S03.1 | One equirect proxy; yaw views as stills, later | Each yaw branch re-resamples the whole frame: 0.73× realtime against 3.46× for proxy-plus-audio. S05 needs one yaw per surviving shot, not four per recording. Set `process.yaw_videos: true` to reproduce the spec's design. |
| One shot per detected scene | Scenes longer than 20 s are divided | 395 of 480 recordings had no detected cut, one of them 29 minutes long. A shot is the unit the timeline chooses between; neither a metric nor a cut survives that length. |
| Gate thresholds as given | Calibrated against the corpus | See [what calibration changed](#what-calibration-against-the-real-corpus-changed). Both rejected the film's spine material. |
| Batch array + ECR + Step Functions (M3, M4, M9) | One rented box running the same CLI | Sized for 500 GB and 2,400 minutes; the actual corpus is 65 GB. Days of build for nothing the resumable CLI does not already give. Revisit if the corpus grows. |
| Glacier IR transition on S03 success (§8.1) | Recommend dropping | Sized for ~500 GB. At the actual ~65 GB it saves about $1.14/month and carries a 90-day minimum billing duration, so on a one- or two-month project it may save nothing at all. |
| `--proxy-only` ingest mode (§9) | Recommend dropping | Its entire rationale was turning 500 GB into 40. At 65 GB it saves about $1/month and 40 minutes of upload, in exchange for giving up cloud conform. |

---

## Troubleshooting

| Symptom | Look at |
|---|---|
| "That config change had no effect" | `nepal doctor` — are you running the checkout or an installed copy? |
| `ModuleNotFoundError` from a `tools/` script | Run through `nepal`, never `python tools/...` |
| A stage prints a full report and changed nothing | The staleness warning above it — every unit was already done |
| The checkpoint table looks wrong | `nepal diagnose` |
| Shots exist that shouldn't | `db.upsert` never removes; something was rebuilt without a delete |
| Metrics all at the same value | A threshold an order of magnitude off — check the percentiles S03.3 logs |
| Proxy has no audio | By design: audio is a separate `.wav` in `work/audio/` |
