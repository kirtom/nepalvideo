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

**Nothing computes on the operator's machine.** It has four cores, eleven
gigabytes and a swap file it lives in; a two-minute test suite makes it
unusable. Stages, tests, probes, renders and API loops run on the GCP box
through `nepal remote` (`up`, `run <nepal args>`, `exec -- <command>`,
`pull`, `down`). Locally: edits, git, `nepal remote`, and queries that
finish in a second. S03.1 takes hours and S03.2 tens of minutes even there;
start them with `nepal remote run` and read the log, never `sleep` waiting.

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
- **Every row of an upsert must carry every column.** `db.upsert` takes its
  columns from the first row. A row that lacked `lat` when the first row had
  it once dropped the column for the whole batch and left 930 video shots
  unpositioned; the upsert now refuses mismatched rows, so write `None`
  rather than omitting a key.
- **A flag written by one sub-step and wiped by another must be derived, not
  stored.** `has_face` was set by S03.6 and lost when S03.2 replaced the
  rows; it now joins `stability`, `has_speech` and `wind` as something the
  gate re-derives from the stored measurement every run.
- **A worktree has no `data/`.** Relative reference paths (`spine.srtm_dir`,
  `spine.geonames_path`) resolve against the project root, and a worktree is
  its own root with no fetched tiles. The first spine re-run from a worktree
  wrote null altitude for every point and collapsed the acts to one day.
  Symlink `data/` from the main checkout before running anything from a
  worktree, and read `nepal doctor` first.
- **A resumable step resumes everything it never saw.** S03.6 resumes on
  `face_score IS NULL`; once photographs became shots, that was 667 rows and
  three hours on this machine, started by a run that only asked for `place`.
  The marking that says "done elsewhere" must happen before the step, not
  after, and a step that can cost hours should say how much it is about to
  resume before it starts.
- **A ghost survives until something deletes it.** The chapter-grouping fix
  landed after S01.2 had run, and three merged "recordings" with 83 surviving
  shots stayed eligible for selection through two draft cuts. `nepal prune`
  exists for this; run it after any change to what the pipeline produces.
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
- **Whisper draws one segment per shot unless asked for words.** With no
  previous text to condition on, a 20 s window comes back as one segment,
  so "cut at an utterance" was "cut at the shot" for 332 transcripts. Ask
  for `word_timestamps` and keep what comes back (`no_speech_prob` too:
  the filter had nothing to read until it was stored).
- **A transcript is not speech until a filter has read it.** Whisper wrote
  subtitle credits over wind, and `has_speech` put them first in their
  acts. Anything that treats a transcript as a claim must check
  `hallucinated` -- or better, read `has_speech`, which the gate now
  derives with the flag folded in.
- **Nobody is named in what leaves the machine.** The chat authors are A,
  B, C by first appearance in the prompt and on every card; the test asserts
  the names are absent. Keep it that way in every stage that asks Claude a
  question. And the bodies, not just the authors: the first live prompt
  carried an `@mention`, a pasted visa email with a full name and a contact
  card with a phone number. `scrub_text` runs on every chat body.
- **A script under `set -x` prints every value it holds.** The bootstrap
  read the API key into a variable and traced it into its log, syslog, the
  serial console and Cloud Logging. Nothing on the box holds the key: the
  profile fetches it from the metadata server at login.
- **`max_tokens` is a backstop, not a budget, and thinking counts against
  it.** A 16,000 cap on a 140k-token prompt was spent entirely on thinking:
  1.11 USD, no answer, and nothing on the ledger because the wrapper raised
  before the stage recorded. The cap is 64,000, and every failed paid call
  records its cost and keeps its partial text before it raises.

## Decisions that are not yours

The three approval gates in the spec are creative judgements — the FOV
confirmation, the shot selection, the draft cut. Surface what is needed to
decide and stop. Same for anything touching AWS spend or credentials.
