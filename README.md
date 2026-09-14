# Nepal Trek Film Pipeline

Turns several hundred GB of raw trek media into a ~20-minute documentary plus a
3-minute short cut, with three human approval gates. Built to the specification
in `docs/spec.md`.

**Status:** Milestone 1 complete — S01 (Probe) and S02 (Spine) both running and validated end to end.

## Build order

Per the spec, the processing logic is built and validated before any AWS
wrapping. Nothing here requires an AWS account yet.

| Milestone | Scope | Status |
|---|---|---|
| 1 | S01 Probe + S02 Spine, locally | **done** |
| 2 | S03 single-clip processing | not started |
| 3 | Containerise | not started |
| 4 | Batch array fan-out | not started |
| 5 | S04 Semantic + S05 Score | not started |
| 6 | The three gates | not started |
| 7 | S06 Assemble + S07 Draft render | not started |
| 8 | S08 Conform + S09 Deliver | not started |
| 9 | Step Functions state machine | not started |

## Install

```bash
apt-get install -y ffmpeg libimage-exiftool-perl     # or: brew install ffmpeg exiftool
pip install -e .                 # Milestone 1 core — no torch, no CUDA
pip install -e '.[music]'        # S02.7 music analysis (librosa, mutagen) -- REQUIRED
pip install -e '.[asr]'          # S02.6 / S03.5 transcription
```

Check the install:

```bash
nepal doctor
```

## Running Milestone 1

Run everything **from the cloned repository**, and leave your media where it
already is — `nepal_data/` is tens of gigabytes and there is no reason to copy
or move it.

```bash
cd nepalvideo
pip install -e .
```

`config/pipeline.yaml` is already pointed at the delivered media:

```yaml
project:
  data_root: /data/projects/nepal_data
  work_root: /data/projects/nepal_work
```

`work_root` sits beside the media rather than inside the repository on purpose:
S03 writes a 540p equirect proxy, four rectilinear yaw views and a 16 kHz audio
track for every recording, which is on the order of 10–15 GB for 4.9 hours of
footage. Move it if that volume is tight.

Relative paths in the config resolve against the **project root** (the folder
containing `config/`), not against your shell's working directory, so the same
config behaves identically wherever you invoke `nepal` from. Confirm what it
resolved to before running anything heavy:

```bash
nepal doctor        # prints every resolved path, and what is missing
```

`data/` holds the fetched reference data (SRTM tiles, GeoNames) inside the repo
and is git-ignored; recreate it any time with `tools/fetch_reference.py`.

Then:

```bash
nepal s01                # manifest, chapters, FOV, clock offsets
nepal fetch-reference    # SRTM tiles + GeoNames gazetteer (once)
nepal s02                # track, altitude, places, telegram, music
nepal report             # the chronological table -- the checkpoint
nepal decisions          # auto-solved values, with confidence
nepal diagnose           # when the checkpoint table looks wrong
```

Run everything through `nepal`, not `python tools/...`. A console script is
bound to the interpreter the package was installed into; a bare `python` picks
up whatever environment happens to be active, which surfaces as a confusing
`ModuleNotFoundError`. `nepal doctor` warns when the two differ. The scripts
under `tools/` are thin shims kept for use before installation.

Which SRTM tiles to fetch depends on where the trek was, so the fetcher works
that out for itself — from the merged GPS track if S02 has run, otherwise from
the photo EXIF that S01 records, otherwise by reading the phone photos
directly. It therefore runs at any point, including before anything else; the
order above just avoids a second pass.

Both stages are resumable: each sub-step records completion, so a re-run redoes
only what is missing. `--force` recomputes.

`nepal s01` is resumable: each sub-step records completion, so a re-run redoes
only what is missing. `--force` recomputes everything; `--skip-fov` and
`--skip-clock` skip the two slow solvers.

### What S01 works out for you

| Decision | How | Fallback |
|---|---|---|
| `fov_deg` | Sweeps 188–206° and picks the FOV whose equirect projection leaves the least discontinuity at the two hemisphere meridians | 193°, flagged for Gate 1 |
| `clock_offset_keller_s` | GCC-PHAT cross-correlation of shared audio between camera and phone takes, median over confident pairs | manual entry at Gate 1, prefilled |
| `clock_offset_kulikov_s` | same | same |
| `camera_has_gps` | direct probe | — |
| `clock_offset_camera_s` | same, two-stage | manual entry at Gate 1 |
| `clock_reference` | whichever device's timestamps agree most closely with the satellite time in `GPSDateTime` | keller |
| Act boundaries | change-point fit on the altitude-vs-day curve | even split by day count |
| Music → act mapping | Hungarian assignment on normalised features, with a callback bonus for Act 5 sharing artist or key with Act 1 | reference palette |

Clock offsets are the highest-consequence values in the project: a wrong one
silently misaligns every downstream join and yields a plausible-looking but
wrong film. They are surfaced at Gate 1 regardless of confidence.

Solving them is two-stage, because one stage cannot span the errors that occur
in practice. A flat battery resets an action camera's clock, and the resulting
error is days wide — thousands of times more than the ±600 s window audio
cross-correlation searches. So candidate offsets are proposed first from
capture *coincidences* (every camera/phone capture pair implies one offset; the
true one is the value many pairs agree on), and audio then arbitrates between
the candidates and refines the winner to sub-second.

**Accuracy ceiling:** device timestamps are whole-second, so clock offsets are
recoverable to roughly ±0.5 s no matter how decisive the correlation is. That
is comfortably inside what geotagging and act assignment need.

## Manual checkpoints

Points where the pipeline stops and asks you to look. These are checks, not
gates — the Step Functions gates come in Milestone 6.

1. **After `nepal s01`** — read the decisions table. Is `fov_deg` plausible
   (188–206, and not the 193 fallback unless the material is poor)? Do the two
   clock offsets look like real clock drift rather than noise? Check
   `work/reports/s01_probe.json` for any chapter-continuity warnings and for
   duplicate files that were collapsed.
2. **After `nepal s02`** — run `nepal report` and read the chronological table.
   This is the checkpoint that decides whether the project is worth continuing.
   Check that: the `pre` row shows enough planning-phase material for Act 1;
   altitude climbs and peaks on the day you remember; the act boundaries land
   where the trek actually changed character; the music assignment reads right
   for each act, and Act 5 echoes Act 1. Any unplaced, unaltituded or unnamed
   assets are counted at the foot of the table with the reason.
3. **Review the music exclusions** in `work/reports/s02_spine.json`. Cyrillic
   detection is exact; the romanised-artist list is best-effort. Correct it in
   either direction with `music.keep_artists` / `music.exclude_artists`.

## Testing

```bash
pytest                   # fast unit tests, no media
pytest -m slow           # end-to-end against generated fixtures (~2 min)
python tools/make_fixtures.py /tmp/fixtures --quick
```

The fixture generator plants known ground truth — a dual-fisheye projection at
a known lens FOV, phone clocks skewed by known amounts, chapter-split takes —
so the tests assert *recovery of the truth*, not merely that the code runs.

## Layout

```
config/pipeline.yaml     every tunable in the spec, in one place
src/nepal/
  config.py              dotted-path config access
  db.py                  SQLite schema (spec section 5) + resumability ledger
  cli.py                 nepal s01 | s02 | decisions | report | doctor
  util/                  ffmpeg/exiftool wrappers, content hashing
  probe/                 S01: manifest, chapters, FOV solver, clock solver
  spine/                 S02: gps, dem, geocode, telegram, music, playlist, acts
  stages/                stage drivers
  reference.py           SRTM tiles and GeoNames gazetteer (nepal fetch-reference)
  diagnose.py            spine diagnostics (nepal diagnose)
tools/make_fixtures.py    synthetic nepal_data/ with ground truth
tools/survey_data.sh      read-only survey of a delivered nepal_data/
tools/survey_camera.sh    camera-clock and music deep dive
tests/                   unit tests + end-to-end
```

Design rule: modules that shell out keep the shelling in a thin wrapper, and
the decision logic that consumes its output stays pure. That is why the FOV
solver, the clock solver and the chapter grouper are unit-testable without any
media present.

## Deviations from the specification

Each of these is a deliberate change with a reason, not an oversight.

| Spec says | Built instead | Why |
|---|---|---|
| Bundle an offline Nominatim extract for reverse geocoding | GeoNames country dump plus a KD-tree | Nominatim needs PostgreSQL, PostGIS and a multi-gigabyte OSM import to name about fifty cluster centroids. A GeoNames dump is a few megabytes of text with no service to run, and it returns villages and peaks rather than postal addresses — which is the question a trek actually asks. |
| Read SRTM tiles (implying a geo stack) | `numpy` directly on `.hgt` | An `.hgt` file is raw big-endian int16 with no header. GDAL and rasterio are a large dependency for one array lookup. Validated against real NASA tiles: Tengboche +2 m, Lukla −10 m, Dingboche −45 m. |
| Camera is the reference clock | The GPS-validated phone is | Photos with a GPS fix also carry satellite time, so `DateTimeOriginal` against `GPSDateTime` is direct evidence of which clock to trust. On this corpus the camera is the one that is provably wrong. |
| Clock offsets from audio cross-correlation over ±600 s | Coincidence voting proposes candidates, then audio arbitrates | The delivered camera's clock is about fourteen days out — roughly two thousand times the ±600 s search window, so the audio stage alone would never have found it. |
| Music features from audio files | Also from a playlist export | The delivered `music/` holds a playlist CSV and no audio. Spotify audio features cover four of the five dimensions the act assignment needs; the fifth, dynamic range, has no playlist analogue and is masked out of the distance rather than approximated. |
| Glacier IR transition on S03 success (§8.1) | Recommend dropping | Sized for ~500 GB. At the actual ~65 GB it saves about $1.14/month and carries a 90-day minimum billing duration, so on a one- or two-month project it may save nothing at all. |
| `--proxy-only` ingest mode (§9) | Recommend dropping | Its entire rationale was turning 500 GB into 40. At 65 GB it saves about $1/month and 40 minutes of upload, in exchange for giving up cloud conform. |
