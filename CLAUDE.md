# Working in this repository

`docs/spec.md` is the source of truth for what gets built; `README.md` explains
the project to a person. This file is for how to work here.

## Environment

- Everything runs through the `nepal` console script, never `python tools/...`.
  A console script is bound to the interpreter the package was installed into;
  a bare `python` picks up whatever environment is active.
- Install editable: `pip install -e '.[vision,music,asr]'`. A non-editable
  install means `git pull` has no effect until you reinstall, and the staleness
  detector reads mtimes of the installed snapshot rather than the working tree.
- `nepal doctor` prints every resolved path and says whether the checkout is
  what actually runs. Run it when anything looks like it is reading the wrong
  file.
- The media (~65 GB) and `work_root` live outside the repository and stay there.
  Never copy or move them.

## Running stages

S03.1 takes hours and S03.2 takes tens of minutes. Start long runs in the
background and poll; never block a turn on them, and never `sleep` waiting.

Every stage is resumable through `stage_units`, so a re-run redoes only what is
missing. `--force` recomputes everything, which for S03 means rebuilding every
proxy — three and a half hours. Prefer `nepal s03 --redo shots,photos` to redo
named sub-steps.

## Conventions

**Every tunable lives in `config/pipeline.yaml`**, reachable by dotted path, so
call sites read like the spec section they implement. No magic numbers in code.
Duplicate keys now raise rather than silently shadowing — a second `process:`
block once deleted the first and cost a 3.5-hour run.

**Modules that shell out keep the shelling in a thin wrapper**, and the decision
logic that consumes the output stays pure. That is why the FOV solver, the clock
solver and the chapter grouper are unit-testable with no media present.

**Comments say why, not what.** The interesting content of this codebase is the
reasoning behind choices that look arbitrary — and most of them were arrived at
by being wrong first.

## Testing

```bash
pytest                   # fast unit tests, no media
pytest -m slow           # end-to-end against ffmpeg-generated fixtures (~2 min)
```

**Verify at the real boundary.** Every metric this pipeline has got wrong was
wrong where Python meets ffmpeg, exiftool or OpenCV, and passed its unit test
because the test fed it a hand-built dict the real scan never produces. When a
change touches what a tool is asked for or what comes back, add a slow test that
runs the actual tool.

## Traps this project has already fallen into

Each of these cost a wrong diagnosis or a wasted multi-hour run.

- **Extension does not tell you what a file is.** `.insv` holds dual-fisheye,
  single-lens and flat 16:9 alike. Classify on frame shape.
- **A tag that is ranked but never requested does not exist.** `exiftool` is
  called with an explicit tag list; the reader can rank `CreationDate` as highly
  as it likes and never see it. `EXIF_TAGS` is derived from `CAPTURE_TAGS` so
  the two cannot drift.
- **`Keys:CreationDate` is the only capture stamp that survives export,
  AirDrop and iCloud** and the only one carrying its own UTC offset.
- **Instrument before theorising.** Four wrong diagnoses of the S03.1 slowdown
  were reasoned from code; per-recording logging plus proxy mtimes answered it
  in one round.
- **`db.upsert` never removes.** Anything a rebuild no longer produces —
  assets, recordings, shots — survives unless it is explicitly deleted.
- **Do not optimise speculatively.** This pipeline runs once. A measured 3x is
  worth taking; a plausible one is not worth the turn.
- **A status whitelist stops matching the moment a later stage promotes a
  row.** S04.1 selected `status = 'candidate'`; S05 promotes its picks to
  `'shortlisted'`, so once a cut existed the stage skipped exactly the shots
  the film was made of. Ask for the complement of the verdict you mean
  (`<> 'rejected'`), which cannot drift as statuses are added.
- **A slot must never claim more footage than its shot has.** The act's
  duration range says what a shot deserves; the shot says what it can give.
  ffmpeg simply stops at the end of the source, so the timeline silently
  over-reports and every number derived from it is wrong.
- **A run measured in hours must checkpoint.** Accumulating in memory and
  writing once at the end means a kill at hour five costs five hours and
  leaves nothing to resume from. Write through a temporary name and rename.

## Decisions that are not yours

The three approval gates in the spec are creative judgements — the FOV
confirmation, the shot selection, the draft cut. Surface what is needed to
decide and stop. Same for anything touching AWS spend or credentials.
