# Nepal Trek Film Pipeline

Turns several hundred GB of raw trek media into a ~20-minute documentary plus a
3-minute short cut, with three human approval gates. Built to the specification
in `docs/spec.md`.

**Status:** Milestone 1 in progress — S01 (Probe) complete and validated.

## Build order

Per the spec, the processing logic is built and validated before any AWS
wrapping. Nothing here requires an AWS account yet.

| Milestone | Scope | Status |
|---|---|---|
| 1 | S01 Probe + S02 Spine, locally | S01 done, S02 next |
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
pip install -e '.[music]'        # S02.7 music analysis
pip install -e '.[asr]'          # S02.6 / S03.5 transcription
```

Check the install:

```bash
nepal doctor
```

## Running Milestone 1

Point `project.data_root` in `config/pipeline.yaml` at your `nepal_data/`
folder, then:

```bash
nepal s01                # manifest, chapter grouping, FOV, clock offsets
nepal decisions          # the auto-solved values, with confidence
```

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

Clock offsets are the highest-consequence values in the project: a wrong one
silently misaligns every downstream join and yields a plausible-looking but
wrong film. They are surfaced at Gate 1 regardless of confidence.

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
2. **After `nepal s02`** — read the chronological table. This is the checkpoint
   that decides whether the project is worth continuing.

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
  spine/                 S02: gps, dem, geocode, telegram, music
  stages/                stage drivers
tools/make_fixtures.py   synthetic nepal_data/ with ground truth
tests/                   unit tests + end-to-end
```

Design rule: modules that shell out keep the shelling in a thin wrapper, and
the decision logic that consumes its output stays pure. That is why the FOV
solver, the clock solver and the chapter grouper are unit-testable without any
media present.
