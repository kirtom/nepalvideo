# The data, and everything done to it

What arrives, what the pipeline makes of it, and what it never touches.
Every number here was measured on the corpus as it stands (2026-09-16) by
querying `work/db/nepal.sqlite` or the filesystem, not estimated.

`docs/spec.md` is the source of truth for what *should* happen;
`docs/STATE.md` says what has actually run. This file is the catalogue in
between: the shape of the material and the transformations applied to it.

---

## 1. What arrives

**65 GB across 1,364 assets**, from five sources that agree about almost
nothing — not container, not clock, not whether they carry GPS.

| Source | Kind | Container | Files | Size | Duration | GPS |
|---|---|---|---:|---:|---:|---:|
| camera | video_flat | mp4 | 64 | 36.9 GB | 52 min | 0 |
| camera | video360 | insv | 32 | 18.8 GB | 45 min | 0 |
| camera | video360 | lrv | 20 | 2.7 GB | 84 min | 0 |
| camera | video_flat | lrv | 69 | 1.4 GB | 55 min | 0 |
| phone_keller | video_flat | mov | 23 | 2.1 GB | 24 min | 23 |
| phone_keller | photo | heic | 339 | 0.8 GB | — | 338 |
| phone_keller | video_flat | mp4 | 42 | 0.2 GB | 2 min | 41 |
| phone_keller | photo | png | 1 | — | — | 0 |
| phone_kulikov | video_flat | mp4 | 32 | 0.4 GB | 17 min | 29 |
| phone_kulikov | video_flat | mov | 273 | 0.4 GB | 10 min | 272 |
| phone_kulikov | photo | jpeg | 282 | 0.3 GB | — | 281 |
| phone_kulikov | photo | jpg | 60 | 0.1 GB | — | 11 |
| telegram | photo | jpg | 77 | — | — | 0 |
| telegram | video_flat | mp4 | 15 | 0.1 GB | 7 min | 0 |
| telegram | document | pdf / json | 6 | — | — | 0 |
| music | audio | mp3 | 26 | 0.1 GB | 116 min | 0 |

Grouped into **480 recordings** (chapter-split files rejoined) and **1,856
shots**.

### The five sources, and what is wrong with each

**`camera`** — an Insta360. Three different things wear the same extensions.
`.insv` holds dual-fisheye, single-lens *and* flat 16:9 alike, so **the
extension tells you nothing and classification is done on frame shape**.
`.lrv` are the camera's own low-resolution proxies. **No GPS at all**
(`camera_has_gps = 0` in `decisions`), and its clock was **18 days out** —
the measured offset is 1,555,080.198 s.

**`phone_keller`** — iPhone. HEIC stills and HEVC video, almost all
geotagged. This is the clock reference (`clock_reference = phone_keller`,
offset 0.0). Carries `Keys:CreationDate`, the **only capture stamp that
survives export, AirDrop and iCloud** and the only one with its own UTC
offset.

**`phone_kulikov`** — a second phone, 647 files, mostly very short. Three
clips are stamped 2025-11-22, which is an export date: every embedded tag
agrees and none carries `Keys:CreationDate`, so their true capture time is
**unrecoverable** and they sit outside every act.

**`telegram`** — a chat export: 1,936 messages, 77 photos, 15 videos and 5
PDFs. Supplies the trip's commentary and the round-video diary entries. Its
timestamps are reliable; its media is re-encoded and low bitrate.

**`music`** — 26 mp3s, 116 minutes. Whatever the operator owned, not a
scored soundtrack.

### Known bad data, documented rather than silently dropped

- **Five camera recordings are black at source** — the camera was recording
  inside a bag. Mean luma ~3/255, 62.8 min, 1.9 GB, 214 shots, every one
  rejected and none in the film.
- **One `awscliv2.zip` was ingested as an asset** (`unknown / other / zip`,
  71 MB). Harmless but wrong: the manifest walks `data_root` and the AWS CLI
  installer was sitting in it.
- **SRTM tiles N28E077, N55E037, N59E030 are absent by design** — Delhi,
  Moscow, St Petersburg. 45 home/transit photos have null altitude.

---

## 2. What is never touched

**The originals are read-only and never leave the machine.** Nothing in the
pipeline writes to `data_root`. Every derived artefact lands in `work_root`,
which is disposable: deleting it costs compute, never material.

---

## 3. Every manipulation, stage by stage

### S01 — Probe: know what was delivered

| Step | Operation | Tool |
|---|---|---|
| `manifest` | sha256 of every file; `asset_id` is the hash | Python |
| | Capture tags read with an **explicit tag list** | exiftool |
| | Stream geometry, duration, fps, codec | ffprobe |
| | Classify `video360 / video_flat / photo / audio / document` **on frame shape, never extension** | ffprobe |
| `chapters` | Rejoin chapter-split files into 480 recordings | pure |
| `gps_check` | Detect that camera GPS is absent | pure |
| `fov` | Solve dual-fisheye field of view by scoring candidate angles | ffmpeg + OpenCV |
| `clock` | Cross-correlate audio between devices to measure clock offset; write `created_at_utc` | ffmpeg + numpy |

Only `created_at_utc` is derived; `created_at` keeps the raw device stamp so
the correction is always reversible.

### S02 — Spine: decide the film's structure. No ML, no GPU.

| Step | Operation | Tool |
|---|---|---|
| `gps_track` | Parse GPX into 1,113 `gps_points` | pure |
| `telegram` | Parse chat export into 1,936 `messages` | pure |
| `geotag` | Assign lat/lon to assets by time against the track; altitude from SRTM tiles | pure + rasterio-free reader |
| `music` | Beat, downbeat and section analysis — 23 tracks, 10,202 beats, 184 sections | librosa |
| `asr` | Transcribe 12 Telegram round videos | faster-whisper `large-v3` |
| `acts` | Segment the trek into five acts by altitude, effort and time | pure |

### S03 — Process: per-clip, and the only expensive stage

| Step | Operation | Tool |
|---|---|---|
| `proxies` | **Dual-fisheye → equirectangular** reprojection at the solved FOV | ffmpeg `v360` |
| | Scale to proxy size, **aspect ratio preserved** (`force_original_aspect_ratio=decrease`) | ffmpeg |
| | Extract 16 kHz mono WAV per recording (479 files, 403 MB) | ffmpeg |
| `shots` | Shot-boundary detection | PySceneDetect |
| `photos` | Turn each photograph into a one-shot row so stills can reach the gate | pure |
| `metrics` | `sharpness` — variance of Laplacian | OpenCV |
| | `exposure_pen` — scene contrast range | OpenCV |
| | `jerk_px` — mean optical-flow jerk; **stability is derived from it at gate time** | OpenCV |
| | `motion_mag` — mean flow magnitude | OpenCV |
| `audio` | `audio_lufs` — integrated loudness | ffmpeg `ebur128` |
| | `wind_lf_share` — low-frequency energy share | numpy |
| | `speech_s` — voiced seconds | silero-VAD (ONNX) |
| `asr` | Transcribe shots with speech — 45k characters of Russian over 537 shots | faster-whisper `large-v3` int8 |
| `faces` | Sample rectilinear views out of the equirect frame at several yaws | OpenCV `remap`, `BORDER_WRAP` |
| | Detect + embed faces (512-d ArcFace) — 1,168 faces | InsightFace `buffalo_l`, onnxruntime CPU |
| `recluster` | Greedy agglomerative clustering + average-linkage merge → 202 clusters | numpy (vectorised) |
| `gate` | Reject soft / too short / shaky / overexposed | pure |

**Every threshold is stored as a measurement and the verdict derived at gate
time**, so moving a threshold is a seconds-long re-run rather than a 38-minute
re-measure.

### S04 — Semantic

| Sub-stage | Operation | Status |
|---|---|---|
| S04.1 | CLIP embedding of the sharpest sampled frame per surviving shot | **built, not yet run** |
| S04.2 | Framing / crop suggestions | **not built** — needs Claude |
| S04.3 | Captions and interest scores | **not built** — needs Claude |

The embedding artefact is a plain `.npy` plus a JSON index, deliberately not
database rows: the spec asks for it to outlive the film as a searchable index
of a personal video library.

### S05 — Score

Three scores per the spec, combined: `score_tech` (sharpness, exposure,
stability, duration fit), `score_sem` (VLM interest, CLIP-act similarity) and
`score_ctx` (faces, speech, new maximum altitude, first-at-place, proximity to
a chat message). **Terms that are missing are dropped and the rest
renormalised** rather than counted as zero.

### S06 — Assemble

MMR selection — `argmax(score − λ·max similarity to what is already chosen)` —
under hard constraints, then laid onto the beat grid. Slot duration is
`min(what the act's range wants, what the shot actually has)`. Exports
OpenTimelineIO and FCPXML 1.9, both written by hand, times quantised to frame
boundaries.

### S07 — Draft render

One ffmpeg invocation, conformed **from the proxies, never the originals**.
Video shots seek at the input (`-ss`/`-t` before `-i`); photographs are held in
the filter graph. Optional shot_id/timecode overlay via `drawtext`.

---

## 4. What lands in `work_root` (4.0 GB)

| Path | Size | Contents |
|---|---:|---|
| `proxies/` | 3.1 GB | 480 `*_eq.mp4` — reprojected, scaled |
| `audio/` | 403 MB | 479 `*.wav` — 16 kHz mono |
| `fov/` | 327 MB | FOV solver frames and thumbnails |
| `gates/` | 195 MB | Gate artefacts, including `gate3/draft.mp4` |
| `clock/` | 41 MB | Audio excerpts for clock correlation |
| `db/` | 12 MB | `nepal.sqlite` — 11 tables |
| `faces/` | 2.4 MB | `embeddings.npy` and the cluster index |
| `transcripts/` | 2.2 MB | Per-shot ASR output |
| `semantic/` | — | `clip.npy` + `clip_index.json` (pending) |
| `timeline.otio`, `.fcpxml` | 196 KB | The editable cut |

## 5. External tools

`ffmpeg` / `ffprobe` (48 + 15 call sites), `exiftool` (36), faster-whisper
(16). Python: OpenCV, numpy, torch, open_clip, InsightFace + onnxruntime,
librosa, soundfile, mutagen, PySceneDetect.

**Note:** this machine's ffmpeg is built **without libfreetype**, so
`drawtext` is unavailable and the draft carries no overlay. It does decode
HEIC.
