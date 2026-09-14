# Nepal Trek Film Pipeline — Build Specification

**Version:** 1.0
**Target implementer:** Claude Code
**Owner:** Kirill Keller
**Status:** ready to build

---

## 0. What this is

An automated pipeline that ingests several hundred GB of raw trek media and produces a finished ~20-minute documentary film plus a 3-minute short cut, with three human approval gates along the way.

The operator runs **one command**. The pipeline pauses three times for a decision, each delivered as a web page with a notification. Everything else is automatic. The output is a rendered video file, not a project to edit.

**Non-goal:** this is not a video editor. It is a selection-and-assembly engine. The operator's creative input is confined to the three gates.

---

## 1. Creative brief

| Attribute | Value |
|---|---|
| Primary output | ~20 min film, 1080p or 4K H.264/H.265 |
| Secondary output | ~3 min short cut, same timeline, higher selection threshold |
| Scope | The whole history: from the first planning message through the trek to the return |
| Tone | Slightly melancholic and epic, **shifting across five acts** — not one sustained mood |
| Narrative spine | Chronological within acts, opened by a cold open (§1.2), with altitude as the dramatic axis |
| Protagonist | The operator (Keller) and his trek partner (Kulikov) |

### 1.1 Five-act structure

| Act | Name | Target duration | Material | Musical character |
|---|---|---|---|---|
| 1 | Planning | **~1.5 min** | Telegram messages, screenshots, maps, gear photos, round video messages | Sparse solo piano. Small, domestic, wistful. Should sound like a city in winter, not mountains. |
| 2 | Approach | 4–5 min | Arrival, low-altitude trail, villages, tea houses | Warmth entering. Strings arriving over piano. |
| 3 | The climb | 6–7 min | High trail, effort, weather, altitude gain | The build. Slow post-rock swell or cold orchestral. Should feel like it costs something. |
| 4 | Highest point | 1.5–2 min | Summit / max altitude reached | Peak swell, **then hard-cut to silence.** 2–3 s of wind and breathing. This is the single most powerful move in the film. |
| 5 | Descent / return | **~5 min** | Coming down, the last days, **Kathmandu, the Delhi layover, the journey home** | Return to the Act 1 piano theme, fuller. The callback is what makes 20 minutes read as a film rather than a montage. |

Act 1 was halved from 2–3 min. Nearly three minutes of planning before anything
happens is a long time to ask for in a twenty-minute film, and the act's material
is its thinnest: seventeen pre-trek clips and sixty-two photographs across two and
a half months. The time moves to Act 5.

Act 5 is no longer only the descent. Descent is an anticlimax -- "walked down" is
not an ending -- but the corpus holds a better one: the return itself. Two hundred
and seventy-six clips from 11-12 May are Kathmandu, the Delhi layover and the
airports home, and three hundred and fifty-four Telegram messages in the "after"
phase are where the reflection lives. The act must reach the flight home, and the
film's last words should come from a message sent after everyone got back.

### 1.2 Cold open (mandatory)

A 15-25 s pre-title sequence drawn from Act 3 or Act 4 -- the pass, or the hardest
moment -- then a hard cut to black and a card reading "three months earlier".

This is the one deliberate break in chronology and it is worth stating plainly as
a deviation. Without it the film opens on ninety seconds of a city in winter, and
asks the audience to wait for a reason to keep watching. Budget comes off the top:
acts are allocated over `target_duration_s - cold_open_s`.

### 1.3 Altitude must be visible

Altitude is named above as the dramatic axis and the film never shows it. The
database holds DEM elevation for 1,083 assets, GeoNames place names, and the full
GPS track; none of it reaches the screen. Required:

- **A place card at each new location**: `Day 6 · Syalagaun · 4,068 m`. Orientation,
  and it makes the climb read as progress rather than as more mountains.
- **A recurring elevation-profile motif** -- a thin line filling as the film
  advances, the pass its obvious apex. Roughly six appearances, ~3 s each.
- **A summit card carrying the number**: `Larkya La — 5,147 m`.

### 1.4 Speech is the spine

A twenty-minute film of landscape under continuous music is a screensaver. What
separates a documentary from a montage is a human thread, and this corpus has
one: 1,936 Telegram messages, 229 of them notable, and round video messages that
are the best narration material in the project.

The original design allotted them one slot, in Act 1. That is a token. Instead:

- S03.5 transcription is **not optional**. Without `faster-whisper` installed the
  film has no voice at all.
- The 8–12 moments where someone *says* something -- a worry, a joke, a decision
  -- are selected first, and the surrounding shots are built around them.
- Sparse voice beats constant narration. Round videos are mostly pre-trek, so
  this mainly carries Act 1; action-camera audio in wind is largely unusable and
  the trek stays mostly wordless. That is the right shape.

### 1.5 Exertion is the story

`new_max_alt` rewards altitude novelty. The dramatically valuable material is
struggle: the slowest hour, the steepest gain, the long unexplained stop. The GPS
track already encodes all three -- speed between fixes, metres gained per hour,
and gaps where nobody moved -- and time of day is free from the timestamps, which
gives the pre-dawn alpine start on summit day and the golden hours either side.

These enter `score_ctx` as first-class terms, and they choose where the
music-out windows of §S07 land.

### 1.6 Tonal constraint (mandatory)

Twenty minutes of unbroken melancholic-epic reads as a perfume advert by minute eight. The assembly **must** place **at least one "levity" shot per act from Act 2 onward** — the bad meal, the argument about the route, someone swearing at a stuck zipper, a stupid joke. Grief and awe only land when something ordinary sits next to them.

These are tagged during the human veto gate and are a hard placement constraint, not a preference.

---

## 2. Source data

### 2.1 Local layout (as delivered)

```
nepal_data/
├── chat_export/                 # Telegram export
│   ├── result.json              # message log
│   ├── files/                   # shared documents (check for GPX, permits, route PDFs)
│   ├── photos/                  # shared photos (EXIF stripped by Telegram)
│   ├── round_video_messages/    # Telegram video circles — talking heads
│   └── video_files/             # shared videos
├── media_from_camera/           # Insta360: .insv / .lrv / .mp4
├── media_from_phones/
│   ├── keller/                  # phone 1 — photos + video, EXIF GPS present
│   └── kulikov/                 # phone 2 — photos + video, EXIF GPS present
└── music/                       # operator's own track selection
```

### 2.2 Target S3 layout

Region: **eu-north-1 (Stockholm)** — nearest to the operator, matters for upload throughput and for remote-desktop latency if a review instance is used.

```
s3://nepal/
├── raw/
│   ├── camera/                  # Standard during S03, then Glacier IR
│   ├── phones/keller/
│   ├── phones/kulikov/
│   ├── chat/
│   └── music/
├── work/
│   ├── db/nepal.sqlite          # pipeline state
│   ├── proxies/                 # 540p equirect proxies
│   ├── views/                   # rectilinear yaw renders
│   ├── frames/                  # sampled JPEGs for embedding/captioning
│   ├── audio/                   # 16 kHz mono WAV
│   ├── gpx/trip.gpx
│   └── music/music_map.json
├── gates/                       # static approval UIs, served via CloudFront
└── deliver/                     # final renders
```

### 2.3 Source characteristics

**Insta360 camera (`media_from_camera/`)**
- `.insv` files are **dual-fisheye**: two H.265 streams side by side, plus an IMU track, plus a `.pb` sidecar carrying lens calibration.
- `.lrv` files are low-resolution proxies of the same dual-fisheye content (roughly 1080p-equivalent). **They already are proxies** — do not re-transcode them.
- `.mp4` files are flat/wide mode, no reprojection needed.
- **No GPS.** Insta360 X-series has no GPS receiver. Location comes only from the GPS Action Remote, which was not used. Verify in S01 but expect nothing.
- Large recordings are chapter-split. Group by the shared recording identifier before scene detection, or every split produces a false shot boundary and the telemetry has a gap.

**Phones (`media_from_phones/{keller,kulikov}/`)**
- EXIF GPS + `DateTimeOriginal` present. **This is the entire geolocation spine for the project.**
- Two devices means **two independent clock offsets** against the camera.
- kulikov's phone is the primary source of shots containing Keller.

**Telegram export (`chat_export/`)**
- `result.json` — timestamped messages. Highest-value input in the project per byte.
- `round_video_messages/` — video circles: face plus voice, timestamped, unperformed. Best narration material available. Process early; there won't be many.
- Telegram **strips EXIF from compressed photos**. Those files have no GPS — recover location by timestamp against the GPX.
- Telegram media is heavily compressed and will fail quality filters tuned for camera footage. It needs **its own quality curve** or Act 1 loses its core material.

---

## 3. Hard technical constraints

These are the things that will silently waste a weekend if not known up front.

| Constraint | Consequence |
|---|---|
| **AWS MediaConvert cannot reproject dual-fisheye.** No `v360` equivalent exists. | The entire 360 half is invisible to it. The managed-video path does not exist for this project. Use own container. |
| **Rekognition Video is order-of-magnitude too expensive.** ~$0.10/min label detection × ~2,400 min = several hundred dollars. | Excluded. Use CLIP embeddings + Bedrock VLM on a filtered subset. |
| **Lambda cannot hold the media.** 15-minute ceiling, 10 GB ephemeral storage. A 4 GB `.insv` will not fit reliably. | Lambda is for orchestration and gates only. All media work runs on Batch. |
| **NAT Gateway is the top cost trap.** ~$33/mo baseline plus per-GB processing on several hundred GB. | Batch runs in a **public subnet with public IP**, plus an **S3 Gateway VPC Endpoint** (free). Never create a NAT Gateway. |
| **Insta360 Studio is GUI-only (Windows/macOS).** | Final high-quality stitching needs `insv-stitch` (open source, IMU stabilisation + rolling shutter from `.pb`) or `insta360-cli-utils` (Docker wrapper around the official Linux Media SDK — SDK access is gated behind an application). Fallback is accepting `v360` seam quality in the finished film. |
| **The upload cannot be avoided.** Several hundred GB must physically leave the operator's machine. | Tune the sync (below). At 100 Mbps up, ~500 GB is roughly half a day. |
| **GPS altitude is unusable at elevation.** | Replace with SRTM DEM lookup against lat/lon. |
| **S3 Standard on the full corpus is the dominant storage cost.** | `raw/camera/` transitions to Glacier Instant Retrieval the moment S03 completes reading it. Glacier IR is ~$0.004/GB/mo vs ~$0.023 Standard; instant retrieval for conform. Avoid Deep Archive — 180-day minimum charge and 12-hour restore will block the edit. |

### 3.1 Upload command

```bash
aws configure set default.s3.max_concurrent_requests 20
aws configure set default.s3.multipart_chunksize 64MB

aws s3 sync ./nepal_data/media_from_camera s3://nepal/raw/camera/   --storage-class STANDARD
aws s3 sync ./nepal_data/media_from_phones s3://nepal/raw/phones/   --storage-class STANDARD
aws s3 sync ./nepal_data/chat_export       s3://nepal/raw/chat/     --storage-class STANDARD
aws s3 sync ./nepal_data/music             s3://nepal/raw/music/    --storage-class STANDARD
```

Skip S3 Transfer Acceleration — $0.04/GB (~$20 on this dataset) for a gain that is not noticeable inside Europe.

---

## 4. Architecture

### 4.1 Components

| Component | Choice | Rationale |
|---|---|---|
| Orchestration | **Step Functions** state machine | Gates need `waitForTaskToken`; visible state; free retries |
| Compute | **AWS Batch on EC2 Spot**, array jobs | Restartable; Spot interruption retries a single array element |
| Instance type | `g4dn.xlarge` | NVDEC for H.265 decode; Whisper and CLIP ride the same GPU |
| Container | **One ECR image** for all stages | ffmpeg, exiftool, PySceneDetect, faster-whisper, open_clip, librosa, numpy, boto3. Each stage is the same image with a different entrypoint. |
| State | **SQLite on S3** (`work/db/nepal.sqlite`) | A few thousand rows. Download at job start, upload at end, with a DynamoDB lock table if concurrency demands it. |
| Vectors | **numpy array in S3** | A few thousand × 768 floats. No vector database is warranted at this scale. |
| Captioning | **Bedrock**, same region | No egress; keeps the only variable cost inside the VPC |
| Gates | **Lambda function URL** + static page on S3/CloudFront | Cheap, mobile-friendly, no server |
| Notification | **SNS → email** (optionally Telegram bot webhook) | Operator gets a link when a gate opens |

### 4.2 State machine

```
S00 Ingest (S3 event on raw/ completion marker)
  ↓
S01 Probe          [Batch, single job]
  ↓
S02 Spine          [Batch, single job]
  ↓
◆ GATE 1 — approve story spine + music assignment   (waitForTaskToken, 24 h timeout → auto-proceed)
  ↓
S03 Process        [Batch, array job over clips]
  ↓
S04 Semantic       [Batch, array job over surviving shots]
  ↓
S05 Score          [Batch, single job]
  ↓
◆ GATE 2 — shot veto                                (waitForTaskToken, NO timeout)
  ↓
S06 Assemble       [Lambda + Bedrock]
  ↓
S07 Draft render   [Batch, single job, proxy resolution]
  ↓
◆ GATE 3 — approve rough cut                        (waitForTaskToken, 48 h timeout → auto-proceed)
  ↓
S08 Conform        [Batch, single job, full resolution]
  ↓
S09 Deliver        [Lambda]
```

Gate 2 has no timeout because it is the primary quality lever. Gates 1 and 3 auto-proceed with defaults so the pipeline never hangs indefinitely.

### 4.3 Gate mechanism

Step Functions `.waitForTaskToken` integration:

1. State machine emits a task token and writes a gate payload to `s3://nepal/gates/<gate_id>/payload.json`.
2. A Lambda generates a static HTML page from the payload and writes it alongside.
3. SNS notifies the operator with the CloudFront URL.
4. The page POSTs the operator's decision plus the task token to a Lambda function URL.
5. That Lambda calls `SendTaskSuccess` with the decision as output, and the machine resumes.
6. On timeout, `SendTaskFailure` is not used — instead the state machine catches `States.Timeout` and routes to a default-decision branch.

---

## 5. Data model

SQLite. Create with this DDL.

```sql
-- Physical files as delivered
CREATE TABLE assets (
  asset_id      TEXT PRIMARY KEY,       -- sha256 of file
  s3_key        TEXT NOT NULL,
  source        TEXT NOT NULL,          -- camera | phone_keller | phone_kulikov | telegram
  kind          TEXT NOT NULL,          -- video360 | video_flat | photo | audio | document
  container     TEXT,                   -- insv | lrv | mp4 | jpg | heic ...
  bytes         INTEGER,
  width         INTEGER,
  height        INTEGER,
  fps           REAL,
  duration_s    REAL,
  created_at    TEXT,                   -- raw device timestamp, NOT corrected
  created_at_utc TEXT,                  -- after clock offset applied
  recording_id  TEXT,                   -- groups chapter-split files
  chapter_index INTEGER,
  has_gps       INTEGER DEFAULT 0,
  lat           REAL,
  lon           REAL,
  alt_dem_m     REAL,
  place_name    TEXT,
  quality_curve TEXT DEFAULT 'camera',  -- camera | phone | telegram
  probe_json    TEXT
);

-- Logical recordings (chapters merged)
CREATE TABLE recordings (
  recording_id  TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  is_360        INTEGER NOT NULL,
  start_utc     TEXT,
  duration_s    REAL,
  asset_count   INTEGER
);

-- Detected shots — the unit of selection
CREATE TABLE shots (
  shot_id       TEXT PRIMARY KEY,
  recording_id  TEXT NOT NULL REFERENCES recordings(recording_id),
  start_s       REAL NOT NULL,          -- offset within recording
  end_s         REAL NOT NULL,
  start_utc     TEXT,
  day_index     INTEGER,
  act           INTEGER,                -- 1..5, assigned in S02/S05
  lat REAL, lon REAL, alt_dem_m REAL, place_name TEXT,
  -- technical signals
  sharpness     REAL,                   -- log Laplacian variance
  exposure_pen  REAL,                   -- 0..1, higher = worse
  stability     REAL,                   -- 0..1, higher = steadier
  motion_mag    REAL,
  audio_lufs    REAL,
  has_speech    INTEGER DEFAULT 0,
  has_face      INTEGER DEFAULT 0,
  face_cluster  TEXT,                   -- keller | kulikov | other | null
  -- 360 framing
  chosen_yaw    REAL,                   -- degrees, null for flat
  view_kind     TEXT,                   -- landscape | subject | null
  -- semantic
  caption       TEXT,
  vlm_interest  REAL,                   -- 1..10
  transcript    TEXT,
  -- scoring
  score_tech    REAL,
  score_sem     REAL,
  score_ctx     REAL,
  score_total   REAL,
  -- human
  vote          INTEGER,                -- 1 keep, -1 kill, null unseen
  tag_levity    INTEGER DEFAULT 0,
  status        TEXT DEFAULT 'candidate' -- candidate | rejected | shortlisted | selected
);

CREATE TABLE gps_points (
  ts_utc  TEXT PRIMARY KEY,
  lat REAL, lon REAL, alt_dem_m REAL, source TEXT
);

CREATE TABLE messages (
  msg_id      TEXT PRIMARY KEY,
  ts_utc      TEXT NOT NULL,
  author      TEXT,
  text        TEXT,
  media_asset TEXT REFERENCES assets(asset_id),
  phase       TEXT,                     -- planning | trek | after
  is_notable  INTEGER DEFAULT 0,
  usable_as_card INTEGER DEFAULT 0
);

CREATE TABLE music_tracks (
  track_id    TEXT PRIMARY KEY,
  s3_key      TEXT,
  title       TEXT,
  duration_s  REAL,
  tempo_bpm   REAL,
  key_est     TEXT,
  energy_mean REAL,
  energy_p95  REAL,
  energy_p10  REAL,
  centroid    REAL,
  onset_rate  REAL,
  assigned_act INTEGER
);

CREATE TABLE music_sections (
  section_id  TEXT PRIMARY KEY,
  track_id    TEXT REFERENCES music_tracks(track_id),
  start_s     REAL, end_s REAL,
  energy      REAL,
  is_swell    INTEGER DEFAULT 0
);

CREATE TABLE beats (
  track_id TEXT, t_s REAL, is_downbeat INTEGER
);

CREATE TABLE timeline (
  slot_index  INTEGER PRIMARY KEY,
  act         INTEGER,
  shot_id     TEXT REFERENCES shots(shot_id),
  msg_id      TEXT REFERENCES messages(msg_id),  -- for caption cards
  t_in        REAL,   -- position on the film timeline
  t_out       REAL,
  src_in      REAL,   -- in-point within the recording
  src_out     REAL,
  yaw         REAL,
  transition  TEXT    -- cut | dissolve | hold | silence
);

CREATE TABLE decisions (
  key TEXT PRIMARY KEY, value TEXT, confidence REAL, method TEXT
);
```

The `decisions` table holds auto-solved parameters (`fov_deg`, `clock_offset_keller_s`, `clock_offset_kulikov_s`, …) so they are inspectable at Gate 1 and overridable by the operator.

---

## 6. Stage specifications

Every stage is idempotent and resumable. Every stage records completion per unit of work in the DB so a Spot interruption re-runs only what is missing.

### S00 — Ingest

**Input:** the `aws s3 sync` commands from §3.1, run by the operator.
**Trigger:** the operator writes `s3://nepal/raw/_COMPLETE` after the syncs finish. An S3 event on that key starts the state machine.

Do not trigger on individual object creation — the sync takes hours and would fire thousands of times.

---

### S01 — Probe

**Purpose:** know exactly what was delivered, and auto-solve the two parameters everything else depends on.

#### S01.1 Manifest

```bash
exiftool -r -j -ee -G1 \
  -FileName -FileType -MIMEType -ImageSize -Duration -VideoFrameRate \
  -CreateDate -DateTimeOriginal -MediaCreateDate \
  -GPSLatitude -GPSLongitude -GPSAltitude -Make -Model \
  /mnt/raw/ > /work/probe/exif.json
```

Populate `assets`. Compute sha256 per file for `asset_id`.

Classify `kind`:
- `.insv`, `.lrv` → `video360`
- `.mp4`, `.mov` from camera or phones → `video_flat`
- `.jpg`, `.heic`, `.png` → `photo`
- anything under `chat_export/` → `source = telegram`, `quality_curve = 'telegram'`

#### S01.2 Chapter grouping

Insta360 splits long recordings. Group by the shared recording identifier in the filename (the segment that stays constant across chapters), ordered by chapter index. Write one row per group into `recordings`, with `duration_s` as the sum and `start_utc` from the first chapter.

**Acceptance:** no `recordings` row has a gap between consecutive chapter end and next chapter start greater than 1 second.

#### S01.3 GPS presence check

```bash
exiftool -ee -G3 -a <sample.insv> | grep -i gps
```

Expect empty. Write `decisions('camera_has_gps', ...)`. If GPS *is* present, S02 can skip the timestamp-interpolation path and geotag directly — branch accordingly.

#### S01.4 Auto-solve fisheye FOV

Replaces "eyeball the seam". Algorithm:

1. Sample 20 frames from across at least 8 different `.lrv` files, preferring frames with high texture (Laplacian variance above the median).
2. For each candidate FOV in `range(188, 208, 2)`:
   ```bash
   ffmpeg -ss <t> -i <clip.lrv> -frames:v 1 \
     -vf "v360=input=dfisheye:output=e:ih_fov=<F>:iv_fov=<F>" \
     -y /work/fov/<clip>_<F>.png
   ```
3. In the equirectangular output of width `W`, the two hemispheres meet at `x = W/4` and `x = 3W/4`. Compute:
   - `seam_grad` = mean absolute horizontal gradient in a 6-px band centred on each seam
   - `local_grad` = mean absolute horizontal gradient in the 60-px bands either side, excluding the seam band
   - `discontinuity = seam_grad / (local_grad + ε)`
4. Score each FOV as the mean discontinuity across both seams, then take the **median across all 20 sampled frames**.
5. `fov_deg` = argmin. Write to `decisions` with `confidence = 1 - (best / second_best)`.

If confidence < 0.05 (no clear winner), fall back to 193 and surface the choice at Gate 1 with the candidate thumbnails.

#### S01.5 Auto-solve clock offsets

Replaces "find a moment captured on both devices". One offset per phone.

**Primary method — audio cross-correlation:**

1. Find candidate pairs: a camera recording and a phone video whose *nominal* timestamps overlap within ±10 minutes and both have audio.
2. Extract 16 kHz mono from both, take up to 120 s from the overlap region.
3. Compute GCC-PHAT cross-correlation over a lag window of ±600 s.
4. `offset = argmax(correlation)`. Confidence = `peak / second_highest_local_peak`.
5. Repeat across all candidate pairs; take the **median offset** of pairs with confidence > 1.5.

**Fallback:** if fewer than 3 confident pairs, mark `confidence = 0` and present at Gate 1 as a manual field, pre-filled with the naive difference of device-reported timestamps.

Write `clock_offset_keller_s` and `clock_offset_kulikov_s` to `decisions`. Apply to every asset as `created_at_utc`.

> This is the highest-consequence value in the pipeline. A wrong offset silently misaligns every join downstream — geotagging, message anchoring, act assignment — and produces a plausible-looking but wrong film. Surface it at Gate 1 regardless of confidence.

**Acceptance:** after applying offsets, at least 90% of camera recordings fall inside the temporal envelope of the merged GPS track.

---

### S02 — Build the story spine

No ML, no GPU. Cheap and fast. This is where the film's structure is determined.

#### S02.1 GPS track

1. Read EXIF GPS + `DateTimeOriginal` from every photo in both phone folders.
2. Merge into one time-sorted point list. On conflict (both phones at the same second), prefer the point with better reported accuracy, else keller.
3. Write `gps_points` and export `work/gpx/trip.gpx`.
4. **Also check `chat_export/files/` for a real `.gpx`** — people share route tracks. If one exists, prefer it and use photo GPS only to fill gaps.

#### S02.2 Geotag everything else

For each camera recording, interpolate lat/lon from `gps_points` at `created_at_utc`. Linear interpolation between the bracketing points; refuse to interpolate across gaps longer than 4 hours (leave null).

#### S02.3 Altitude

Look up SRTM DEM at each lat/lon. Do **not** use GPS altitude — it is unusable at elevation. Bundle the relevant SRTM tiles in the container image rather than calling an API per point.

#### S02.4 Reverse geocode

Batch reverse-geocode distinct locations (cluster to ~500 m) to place names. Use a bundled offline Nominatim extract for Nepal — one-time, free, no rate limits.

#### S02.5 Parse Telegram

Read `result.json`. For each message write a `messages` row.

Classify `phase`:
- `planning` — before the first GPS point in the track
- `trek` — within the track's time envelope
- `after` — following the last GPS point

Mark `is_notable` where a message sits within 20 minutes of a local maximum in altitude gain, or contains a media attachment, or is unusually long relative to the thread median.

Mark `usable_as_card` for short, self-contained, quotable messages — these become Act 1's on-screen text.

Link `chat_export/` media files to `assets` via message references, inheriting the message timestamp as `created_at_utc` (Telegram compressed media has no EXIF).

#### S02.6 Transcribe round video messages

```bash
whisper --model large-v3 --language ru --output_format json <file>
```

Run over the whole `round_video_messages/` folder and any voice notes. No VAD gating needed — the folder is small and entirely speech. These are the best narration material in the project; process them here rather than waiting for S03.

#### S02.7 Music analysis

For every track in `raw/music/`:

```python
y, sr = librosa.load(path)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
rms = librosa.feature.rms(y=y)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
onset = librosa.onset.onset_strength(y=y, sr=sr)
key = estimate_key(librosa.feature.chroma_cqt(y=y, sr=sr))
bounds = librosa.segment.agglomerative(librosa.feature.mfcc(y=y, sr=sr), k=8)
```

Write `music_tracks`, `music_sections`, `beats`. Mark a section `is_swell` where its energy exceeds the track's p75 **and** rises monotonically from the previous section.

**Act assignment.** Build a feature vector per track: `[energy_mean, energy_p95 - energy_p10, centroid, onset_rate, tempo]`, each z-normalised across the library. Define target vectors per act:

| Act | energy | dyn. range | centroid | onset rate | tempo |
|---|---|---|---|---|---|
| 1 Planning | low | low | low | low | low |
| 2 Approach | mid | mid | mid | mid | mid |
| 3 Climb | mid-high | **high** | mid | mid | mid |
| 4 Summit | **highest** | high | high | high | any |
| 5 Descent | low-mid | mid | low | low | low |

Solve as a bipartite assignment (Hungarian) minimising Euclidean distance. **Add a bonus for Act 5 sharing key or artist with the Act 1 track** — the callback is a design requirement, not an accident.

If the library is too small or a poor fit, fall back to the reference palette in §11 and surface the choice at Gate 1.

#### S02.8 Emit the music map

`work/music/music_map.json`:

```json
{
  "total_duration_s": 1200,
  "acts": [
    {
      "act": 1,
      "track_id": "…",
      "t_start": 0.0,
      "t_end": 168.0,
      "swells": [42.5, 131.0],
      "beat_grid": [0.0, 1.94, 3.88, "…"],
      "downbeats": [0.0, 7.76, 15.52, "…"]
    }
  ],
  "silence_window": { "t_start": 812.0, "t_end": 815.5 }
}
```

`silence_window` is the mandatory hard cut to silence after the Act 4 peak.

**Acceptance:** act durations sum to within ±30 s of the 1,200 s target; every act has at least one swell timestamp.

---

### ◆ GATE 1 — Approve the story spine

**Payload:** chronological table (day, place, altitude, clip count, photo count, message count), an interactive route map rendered from the GPX, the proposed act boundaries as draggable day-range markers, the proposed music assignment with 30-second audio previews, and the auto-solved `decisions` values with confidence scores.

**Operator can:** shift act boundaries, swap a track between acts, override `fov_deg`, override either clock offset.

**Returns:**
```json
{
  "act_boundaries": [{"act": 1, "start_utc": "…", "end_utc": "…"}],
  "music_assignment": {"1": "track_id", "…": "…"},
  "overrides": {"fov_deg": 193, "clock_offset_keller_s": -37}
}
```

**Timeout:** 24 h → proceed with the auto-computed values.

---

### S03 — Process and extract cheap signals

Batch **array job**, one element per `recordings` row. Single pass over each recording — never re-read the source for a second metric.

#### S03.1 Reprojection and views (360 only)

One ffmpeg invocation producing all outputs, chaining directly from dual-fisheye to each target (never via an intermediate file):

```bash
ffmpeg -hwaccel cuda -i <input> \
  -filter_complex "\
    [0:v]split=6[eq][v0][v90][v180][v270][aud]; \
    [eq]v360=input=dfisheye:output=e:ih_fov=${FOV}:iv_fov=${FOV},scale=1024:512[eqout]; \
    [v0]v360=input=dfisheye:output=rectilinear:ih_fov=${FOV}:iv_fov=${FOV}:yaw=0:h_fov=100:v_fov=70,scale=960:540[y0]; \
    [v90]v360=input=dfisheye:output=rectilinear:ih_fov=${FOV}:iv_fov=${FOV}:yaw=90:h_fov=100:v_fov=70,scale=960:540[y90]; \
    [v180]v360=input=dfisheye:output=rectilinear:ih_fov=${FOV}:iv_fov=${FOV}:yaw=180:h_fov=100:v_fov=70,scale=960:540[y180]; \
    [v270]v360=input=dfisheye:output=rectilinear:ih_fov=${FOV}:iv_fov=${FOV}:yaw=270:h_fov=100:v_fov=70,scale=960:540[y270]" \
  -map "[eqout]" -c:v h264_nvenc -b:v 2M  /work/proxies/<rid>_eq.mp4 \
  -map "[y0]"    -c:v h264_nvenc -b:v 1M  /work/views/<rid>_y0.mp4 \
  -map "[y90]"   -c:v h264_nvenc -b:v 1M  /work/views/<rid>_y90.mp4 \
  -map "[y180]"  -c:v h264_nvenc -b:v 1M  /work/views/<rid>_y180.mp4 \
  -map "[y270]"  -c:v h264_nvenc -b:v 1M  /work/views/<rid>_y270.mp4 \
  -map 0:a -ac 1 -ar 16000 -c:a pcm_s16le /work/audio/<rid>.wav
```

Prefer `.lrv` as input where it exists — it is already a proxy and saves the decode of a 5.7K H.265 stream. Use `.insv` only if no `.lrv` is present.

Flat `.mp4` recordings skip the `v360` filters entirely and produce a single 540p proxy plus audio.

#### S03.2 Shot detection

PySceneDetect `ContentDetector` on the equirect proxy (or the flat proxy). Threshold 27, minimum shot length 1.5 s.

**Run per `recording_id`, not per file** — chapter joins must not produce boundaries.

Write `shots` rows.

#### S03.3 Technical metrics per shot

Sample 5 frames evenly within each shot from the equirect proxy:

- `sharpness` = `log(var(Laplacian(gray)))`, median across samples
- `exposure_pen` = fraction of pixels above 250 or below 5, median across samples
- `motion_mag` = mean magnitude of Farnebäck optical flow between consecutive sampled frames
- `stability` = `1 / (1 + var(motion vectors))`; where IMU is available, prefer `1 / (1 + mean(|jerk|))` from the gyro track

#### S03.4 Audio

- `audio_lufs` = EBU R128 integrated loudness per shot (`ffmpeg -af ebur128`)
- Wind detection: flag shots whose spectral energy below 200 Hz exceeds 80% of total
- `has_speech` = silero-VAD over the shot's audio window

#### S03.5 Transcription

Run faster-whisper **only** on shots where `has_speech = 1`. On action-camera audio in wind this is typically under 10% of runtime — but it contains most of the human moments.

#### S03.6 Faces

InsightFace over the sampled frames of the `y0/y90/y180/y270` views (not the equirect — faces are distorted there). Cluster embeddings across the whole corpus, then label the two largest clusters `keller` and `kulikov` — confirm the labelling at Gate 2 by showing one representative face per cluster.

Set `has_face` and `face_cluster` on the shot, recording which yaw view the face appeared in.

#### S03.7 Quality gate

Reject shots where, **against the curve appropriate to the asset's `quality_curve`**:

| Rule | camera | phone | telegram |
|---|---|---|---|
| min sharpness (log Laplacian) | 4.0 | 3.5 | **2.0** |
| max exposure penalty | 0.15 | 0.20 | **0.35** |
| min stability | 0.35 | 0.30 | **0.15** |
| min duration | 1.5 s | 1.5 s | 1.0 s |

Set `status = 'rejected'`. Expect roughly 70% rejection on camera material. Telegram material must not be judged on the camera curve or Act 1 loses its core content.

#### S03.8 Storage transition

On successful completion of the whole array job, transition `s3://nepal/raw/camera/` to Glacier Instant Retrieval.

---

### S04 — Semantic layer

Batch array job over surviving shots only.

#### S04.1 Embeddings

One representative frame per shot (the sharpest sampled frame), through SigLIP or CLIP ViT-L. Store in a single numpy `.npy` in `work/` with a parallel `shot_id` index. This artefact is reusable as a personal video-library search index beyond this project — build it to outlive the film.

#### S04.2 360 framing decision

For each 360 shot, send the four yaw views as a 2×2 contact sheet to Bedrock and ask for:
- which view is the most compelling composition (returns yaw)
- whether the chosen view contains a person as the subject (returns `view_kind`)

Where `has_face = 1`, emit **two** candidate rows for the shot: the landscape view (best-composition yaw) and the subject view (the yaw where the face was detected). Both compete independently in scoring.

#### S04.3 Captioning

Send 4 frames per surviving shot plus the shot's transcript snippet, if any.

**Inject the Telegram vocabulary into the prompt** — place names, "Keller", "Kulikov", gear names, recurring jokes, extracted from `messages`. Without this the captions come back in generic model English and retrieval stops working in the operator's own language.

Return strict JSON:
```json
{"caption": "...", "interest": 7, "subjects": ["person","ridge"], "time_of_day": "dawn", "levity": false}
```

Write `caption`, `vlm_interest`.

**Cost control:** this stage is the entire variable cost of the project. Cap the number of captioned shots via a `MAX_CAPTION_SHOTS` parameter (default 2,500). If more survive the quality gate, rank by `score_tech` first and caption only the top N.

---

### S05 — Score and shortlist

```
score_tech = 0.35·norm(sharpness)
           + 0.25·(1 − exposure_pen)
           + 0.30·stability
           + 0.10·duration_fitness          # peaks at 4–8 s, falls off outside

score_sem  = 0.60·(vlm_interest / 10)
           + 0.40·clip_similarity_to_act_prompt

score_ctx  = 0.25·has_face
           + 0.20·has_speech
           + 0.20·new_max_altitude          # first shot at a new altitude record
           + 0.20·first_shot_at_place
           + 0.15·message_proximity         # a message was sent within 20 min

score_total = 0.40·score_tech + 0.35·score_sem + 0.25·score_ctx
```

`clip_similarity_to_act_prompt` compares the shot embedding against act-specific text prompts, e.g. Act 3: *"a steep high-altitude trail, thin air, effort, exposed ridge, cold light"*.

Assign `act` per shot from `start_utc` against the Gate 1 act boundaries.

Take the top **400** by `score_total`, stratified so no act contributes fewer than 40 candidates. Set `status = 'shortlisted'`.

---

### ◆ GATE 2 — Shot veto

The primary quality lever. **No timeout.**

**Payload:** a contact sheet of the 400 shortlisted shots, grouped by act, each showing the chosen frame at the chosen yaw, its caption, its day and place, and its score.

**Operator can:**
- keep / kill each shot (default: keep)
- flip a 360 shot's framing between the landscape and subject view
- tag a shot **levity**
- confirm the two face-cluster labels

Design for a phone. Large tap targets, swipe or two-button, progress indicator. Target completion time: 45 minutes.

**Returns:**
```json
{
  "votes": {"shot_id": 1, "shot_id_2": -1},
  "yaw_overrides": {"shot_id": 270},
  "levity": ["shot_id", "…"],
  "face_labels": {"cluster_0": "keller", "cluster_1": "kulikov"}
}
```

---

### S06 — Assemble

Lambda plus a Bedrock call. The corpus is now a few hundred rows of text and fits in context comfortably — this is the part that must **not** be attempted with a video model.

**Input to the model:** the surviving shots as a table (shot_id, act, day, place, altitude, caption, transcript snippet, score, tags), the music map from §S02.8, and the constraints below.

**Hard constraints:**

| Constraint | Value |
|---|---|
| Total duration | 1,200 s ± 30 s |
| Shot count | 150–200 |
| Chronology | Strictly non-decreasing within each act |
| Act durations | As allocated in `music_map.json` |
| Subject shots | ≥ 1 containing a person per 40 s of runtime |
| Levity | ≥ 1 levity-tagged shot per act, Acts 2–5 |
| Location diversity | ≤ 3 shots from the same `place_name` per act |
| Cut points | Snapped to the nearest beat; act transitions snapped to a downbeat |
| Cold open | 15–25 s from Act 3 or 4 at the head, then a title card. The only break in chronology |
| Act 1 | Message caption cards interleaved; ≥ 1 round-video-message clip |
| Act 4 | Peak shot lands on the highest swell, followed by `silence_window` — a single held shot with audio faded to wind only |
| Act 5 | Must open on a shot echoing an Act 1 composition where one exists, and **must reach the journey home** — ≥ 3 slots from Kathmandu, the Delhi layover or the flight, and a closing card from an "after"-phase message |
| Speech | Every transcribed moment that survives Gate 2 is placed. Speech is the spine, not a bonus — see §1.6 |
| Natural sound | 3–5 music-out windows of 10–20 s, placed at the highest-exertion moments — see §1.7 |
| Place cards | One per new `place_name`, carrying day and altitude |

**Shot duration model** (before beat snapping):

| Act | Base duration |
|---|---|
| 1 | 3–5 s |
| 2 | 4–6 s |
| 3 | 3–8 s, shorter as energy rises |
| 4 | 2–4 s at peak, then one 6–10 s held shot |
| 5 | 5–8 s |

**Selection algorithm.** Do not let the model free-form the whole timeline. Instead:

1. Per act, compute the slot budget from the music map.
2. Greedy MMR fill: at each step select `argmax(score_total − λ · max_cosine_similarity(candidate, already_selected))`, λ = 0.3.
3. Apply hard constraints as filters at each step; if a constraint cannot be satisfied, relax diversity before relaxing chronology.
4. Pass the resulting ordered list to Bedrock **only** for ordering refinement within act and for choosing the caption-card placements — not for the selection itself.

**Output:** `timeline` rows, plus an OpenTimelineIO JSON at `work/timeline.otio` and an FCPXML export at `work/timeline.fcpxml` for optional manual refinement.

---

### S07 — Draft render

Batch job. Conform from the **proxies**, not the originals. 960×540, H.264, music bed mixed at −14 LUFS, original audio ducked to −28 LUFS under music except where `has_speech = 1` (duck music to −22 instead).

**Music-out windows.** Music at −14 with everything under it at −28 means the
location sound is inaudible for the whole runtime, which throws away the most
visceral material the trek produced: breathing at five thousand metres, wind,
boots on scree, a river. Three to five windows of 10–20 s carry **no music at
all** — original audio at full level — placed at the highest-exertion moments
S05 identifies. The Act 4 silence lengthens from 3 s to 5 s: a hard cut to
silence needs room to land.

Burn in shot_id and timecode as a small overlay so Gate 3 feedback can reference specific moments.

Write to `s3://nepal/gates/gate3/draft.mp4`.

---

### ◆ GATE 3 — Approve the rough cut

**Payload:** the draft video, plus a shot list with timestamps.

**Operator can:** approve, or mark timestamps to remove. Marked shots are set `vote = -1` and the machine loops back to S06 for reassembly — not back to S04, so no additional model cost is incurred.

**Returns:**
```json
{"approved": true}
```
or
```json
{"approved": false, "remove_shot_ids": ["…"], "notes": "act 3 drags around 9:00"}
```

**Timeout:** 48 h → approve.

Cap the reassembly loop at 3 iterations, then proceed regardless.

---

### S08 — Conform

1. Restore the selected recordings from Glacier IR (instant; only the ~120 recordings actually used).
2. For each selected 360 shot, run **proper stitching** — `insv-stitch` or `insta360-cli-utils` — at full resolution with IMU stabilisation and rolling-shutter correction, then reframe to the chosen yaw at output resolution.
   - If neither stitcher is available, fall back to `v360` at full resolution and accept the seam quality. Record which path was taken.
3. Conform the timeline from full-resolution sources.
4. Grade: apply a single normalisation LUT per source type — the Insta360, the two phones and Telegram media will not match natively.
5. Render the route-map animation from the GPX as an overlay for act transitions.
6. Render Act 1 caption cards from `messages` where `usable_as_card = 1`.
7. Final audio mix.

---

### S09 — Deliver

Render both cuts from the same `timeline`:

- **`nepal_20min.mp4`** — full timeline
- **`nepal_3min.mp4`** — re-run S06's MMR fill with a 180 s budget and the same constraints proportionally scaled; the higher threshold naturally selects the strongest shots

Write to `s3://nepal/deliver/`, generate presigned URLs, notify via SNS.

---

## 7. Auto-solved decisions

Every one of these replaces a manual step. All are written to `decisions` with a confidence score and are overridable at Gate 1.

| Decision | Method | Fallback |
|---|---|---|
| `fov_deg` | Seam-discontinuity minimisation over 20 sampled frames (§S01.4) | 193, surfaced at Gate 1 with thumbnails |
| `clock_offset_keller_s` | GCC-PHAT audio cross-correlation, median over confident pairs (§S01.5) | Manual entry at Gate 1 |
| `clock_offset_kulikov_s` | Same | Manual entry at Gate 1 |
| Act boundaries | Altitude profile segmentation + message phase classification | Even split by day count |
| Music → act mapping | Hungarian assignment on normalised audio features (§S02.7) | Reference palette (§11) |
| 360 yaw per shot | VLM ranking of four candidate views (§S04.2) | Highest-saliency view |
| Face cluster labels | Two largest clusters, confirmed at Gate 2 | Unlabelled |
| Shot selection | MMR greedy under hard constraints (§S06) | — |

---

## 8. Cost model

Region eu-north-1. Verify against current pricing — these move.

| Item | Estimate |
|---|---|
| S3 Standard, ~500 GB, one month during processing | ~$12 |
| Glacier IR after S03 transition | ~$2/mo |
| Batch Spot `g4dn.xlarge`, S03 + S04 | $5–10 |
| Bedrock captioning, 2,500 shots × 4 frames | $20–40 |
| Lambda, Step Functions, SNS, CloudFront | < $1 |
| Optional NICE DCV review instance | $10–20 |
| **Total** | **~$50–85** |

### 8.1 Cost guardrails — implement these on day one

1. **AWS Budgets alarm** at the operator's ceiling, alerting at 50% and 80%.
2. **No NAT Gateway.** Batch runs in a public subnet with a public IP; an S3 Gateway VPC Endpoint (free) handles all S3 traffic. This is the single largest avoidable cost.
3. **CloudWatch log retention set to 7 days** at log-group creation. The default is never-expire.
4. **`MAX_CAPTION_SHOTS` parameter**, default 2,500, enforced in S04. This is the only unbounded cost in the system.
5. **Spot only** for Batch compute, with `attemptDurationSeconds` and retry on interruption.
6. **Glacier IR transition** fires automatically on S03 success, not on a 30-day lifecycle rule.

### 8.2 Free tier

Do not plan around it. S3's always-free allowance is 5 GB — irrelevant against this dataset. If the AWS account postdates July 2025 it is on the credit model ($100–200, six months) and Free Plan accounts are restricted from some services; move to a Paid Plan with a Budgets alarm rather than risking the pipeline being cut off mid-run.

---

## 9. The reduced-upload variant

If the operator will accept **one** local step at the end, the economics improve sharply:

- Upload only the `.lrv` proxies, phone media, chat export and music — roughly 40 GB instead of 500.
- Run S01 through S07 entirely in the cloud, unchanged.
- Skip S08's Glacier restore; pull down `timeline.fcpxml` and conform locally from the originals still on disk.

Trade-off: loses the ability to grade in the cloud, and the operator must have Resolve or ffmpeg locally. Saves the Standard-tier storage bill almost entirely and cuts upload time by 8×. Everything architecturally interesting stays cloud-native.

Implement this as a `--proxy-only` ingest mode rather than a separate pipeline.

---

## 10. Build order

Do not build the state machine first. Build the processing logic, then wrap it.

**Milestone 1 — Probe and spine, running locally**
S01 + S02 against the local `nepal_data/` folder. No AWS, no containers. Output: the chronological table and `music_map.json`.
*This is the checkpoint that decides whether the project is worth continuing.* Read the table before building anything else.

**Milestone 2 — Single-clip processing**
S03 for one recording, end to end, locally. Verify the reprojection, the four yaw views, the shot boundaries and the metrics by eye. Get `fov_deg` right here.

**Milestone 3 — Containerise**
Build the ECR image. Run Milestone 2's code inside it against S3 paths. One clip.

**Milestone 4 — Batch array**
Fan S03 out across the corpus. Verify resumability by killing a job mid-run and re-running.

**Milestone 5 — Semantic and scoring**
S04 + S05. Start with `MAX_CAPTION_SHOTS=50` to validate the prompt and the JSON contract before spending real money.

**Milestone 6 — Gates**
The three Lambda + static-page gates and the `waitForTaskToken` wiring. Test with fabricated payloads.

**Milestone 7 — Assembly and draft render**
S06 + S07. Iterate on the constraint weights against the draft output — this is where the film's quality actually lives.

**Milestone 8 — Conform and deliver**
S08 + S09, including the stitcher decision.

**Milestone 9 — State machine**
Wire the whole thing into Step Functions last, once every stage runs standalone.

---

## 11. Reference music palette

Used if the operator's `music/` library does not fit the act targets. All are named for register, not as a requirement.

| Act | Register | Reference |
|---|---|---|
| 1 Planning | Sparse solo piano, domestic, wistful | Nils Frahm — *Says*, *Ambre* |
| 2 Approach | Strings arriving over piano, warmth | Ólafur Arnalds — *Near Light*, *Ljósið* |
| 3 Climb | Slow post-rock swell, or cold orchestral | Hammock; This Will Destroy You; Jóhann Jóhannsson |
| 4 Summit | Peak swell → hard cut to silence | Any Act 3 track's climax |
| 5 Descent | Act 1 theme returning, fuller | Same artist/key as Act 1 |

**Licensing.** All of the above will be claimed on YouTube. Cleared substitutes in the same register: **Scott Buckley** (free, CC-BY — *Discovery*, *Snowfall*, *Legionnaire*), Kai Engel and Chris Zabriskie for the sparse-piano slot. Alternatively a month of Musicbed or Artlist.

Set a `MUSIC_LICENSE_MODE` config value (`personal` | `cleared`) so the act assignment can be restricted to the cleared library when the destination is public.

---

## 12. Open questions

Resolve these before or during Milestone 1.

1. **Do the `.insv` originals exist, or only `.lrv`?** If LRV-only, the 360 half caps around 1080p and §S08's stitching step is moot. This changes what film is possible — check first.
2. **Does `chat_export/files/` contain a GPX or route file?** Would replace §S02.1's photo-derived track with something better.
3. **Which language for Whisper?** Assumed Russian for the round video messages; confirm.
4. **Insta360 Media SDK access** — worth applying for early if full-quality stitching matters, since approval takes time. Decide the fallback now.
5. **Destination** — YouTube, private, or both. Determines `MUSIC_LICENSE_MODE`.
6. **Kulikov's consent** for his messages and face appearing on screen. Not a technical blocker, but Act 1 is built substantially from his words.

---

## 13. Summary of the decision record

Everything below was decided during design and should not be relitigated without reason.

- **Not a video model on raw footage.** Frontier models handle hour-scale video, but 40+ hours is unaffordable and 90% of it is junk. Cheap deterministic filtering first, expensive semantic work only on survivors.
- **The output of assembly is a timeline, not a render.** The draft render exists for Gate 3 review; FCPXML/OTIO is always exported so manual refinement stays possible.
- **Photos are the geolocation spine.** The camera has no GPS; this is the load-bearing integration in the whole design.
- **The Telegram export is the highest-value input per byte.** Timeline anchors, vocabulary, three-act structure, and on-screen text for Act 1.
- **The 360 camera solves the protagonist problem.** It filmed the operator continuously; the subject view must be a first-class candidate, not an afterthought.
- **Three gates, not a runbook.** One trigger, three decisions, finished film. Gate 2 has no timeout because it is where quality is actually determined.
- **Simplicity over managed services.** SQLite over Aurora, numpy over a vector DB, one container over five services. At a few thousand rows the fancy architecture costs more than it returns.
