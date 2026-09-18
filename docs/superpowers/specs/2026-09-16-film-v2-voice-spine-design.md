# Film v2 — voice spine, two-track timeline, motion, and a real ending

**Status:** design, awaiting operator review
**Amends:** `docs/spec.md` §1, §5, §6 (S02.7–S02.8, S04–S09), §8
**Date:** 2026-09-16

---

## 0. Why this document exists

The first draft cut (`work/gates/gate3/draft.mp4`, 18.7 min) was inspected on
2026-09-16 against the code, the corpus and the database. The finding was not
"the Claude stages are blocked", it was that the parts of the design that make a
film out of a montage are either unbuilt or built and never called, and that the
unit of selection is wrong for the material.

What was measured, in the order it costs the film:

| Finding | Evidence |
|---|---|
| The draft is silent | one video stream, no audio; encoded at 120 fps |
| No title, cold open, cards, map, silence beat or credits | every `timeline` row has `transition = NULL`, `msg_id = NULL` |
| Speech is chopped mid-sentence and often not even inside the slot | slots take the first 4–7 s of a 20 s chunk; 11 runs of 3–9 consecutive slots from one recording |
| Whisper hallucinations counted as speech and placed first | "Субтитры сделал DimaTorzok" on 33 shots, "Продолжение следует" on 23 |
| Faces never reach selection | `has_face = 0` on all 1,856 rows although 347 carry a `face_cluster` |
| Video has no place, altitude or day | 0 of 930 surviving video shots positioned; photos are — so `score_ctx` favours stills |
| Stills then videos | Act 2 is 35 stills of 40 slots; the photo budget (`select.py`) is never called |
| Trek footage in the Planning act | a ghost recording `phone_kulikov_IMG` (302 clips merged, 26 min) survives `db.upsert`; 9 of its slots sit in Act 1 |
| Portrait clips pillarboxed, 360 shown as raw equirect, stills static | 367 of 458 flat recordings are portrait; S04.2 framing unbuilt |
| Flat rhythm | slot length is score-scaled inside a 2–3 s band; scores span 0.52–0.66 |
| Music assigned blind | the feature solve put *Born Slippy* on "sparse solo piano, a city in winter" |
| Subject, levity and effort constraints exist and are never invoked | `assemble.needs_subject`, `missing_levity`, `spine/effort.py` have no callers |

Captions and CLIP embeddings would fix repetition and ranking. They would not
fix the unit, the rhythm, or what is on screen. This design does.

---

## 1. The film (amended brief)

Everything in spec §1 stands except where restated here.

### 1.1 Six acts, not five

"Walked down" was already rejected as an ending; the return through Kathmandu,
the Delhi layover and the flight home was folded into Act 5 and then starved:
the descent alone took the budget. The corpus has two days of descent, three in
Kathmandu (11 to 13 May, including 36 minutes of camera on the 13th), a day in
Delhi, and the operator is adding Delhi footage. That is two acts.

| Act | Name | Target | Boundary | Musical character |
|---|---|---|---|---|
| 0 | Opening + cold open | 40 s | the rise and the flyover (§13.8), then 15–25 s from Act 3 or 4, hard cut to black, "three months earlier" | the summit cue, cut dead |
| 1 | Planning | 80 s | first planning message → departure | small, domestic, wistful |
| 2 | Approach | 240 s | arrival → where the ascent steepens | warmth entering |
| 3 | The climb | 360 s | → summit push | the build; costs something |
| 4 | Highest point | 110 s | the contiguous summit band | peak swell → **5 s silence** |
| 5 | Descent | 170 s | summit → first fix inside `spine.return_radius_km` of Kathmandu | release; the Act 1 theme returning |
| 6 | Return | 200 s | Kathmandu → last message in the "after" phase | fuller callback; the last words are a message sent after everyone got home |

Sum 1,200 s plus credits (60–120 s, outside the runtime). `acts` in
`config/pipeline.yaml` gains the sixth row; `spine/acts.py` gains the return
boundary, found as the first GPS fix after the summit within
`spine.return_radius_km` (default 30) of the geocoded `spine.return_place`
(default `Kathmandu`). Act 6 grows when the Delhi material lands; its `max_s`
allows 400. Where a Strava activity exists (§13.1) its start and end are the
day's boundaries and its name is the day's title.

### 1.2 The unit is a moment, not a chunk

The spec's shot is whatever PySceneDetect or the 20 s cap produced. That stays
as the unit of *measurement*. The unit of *selection and assembly* becomes:

- **a beat**: one complete utterance (or a run of them) with its own in/out
  points on the recording's clock, taken from whisper segment times, not from
  the chunk boundary;
- **a picture slot**: a span of one source, any length the rhythm asks for,
  which may sit under a beat's audio or carry its own.

Audio and picture are separate tracks with separate sources. This is the change
that makes everything else possible.

### 1.3 Speech is chosen by reading, not by scoring

`speech_first` ordering plus `has_speech` weight is replaced by a **beat
sheet**: Claude reads every transcript with timing, the whole Telegram thread,
and the day table, and returns the 10–16 spoken moments that carry the story,
the Act 1 quotes, the closing line, and the title. Selection builds around
those. Section 4 has the contract.

### 1.4 Rhythm comes from the music

Slot length follows the section energy of the cue under it, not the shot's
score. Quiet passages hold; swells cut on downbeats; each act's peak gets one
burst montage. Section 5.4.

### 1.5 Everything moves

No static frame is delivered: stills get a Ken Burns move, 360 shots get a yaw
drift, portrait clips get a blurred fill or a face-tracked crop, and two phones
that filmed the same minute share the frame. Section 6.

### 1.6 Altitude and time are always visible

Two corner widgets run for the whole trek: a route map with the track filling
in and the current altitude, and a day counter. Place cards at every new place,
an elevation-profile motif at act transitions, and one statistics card per act.
Section 6.4.

### 1.7 The chat is on screen, anonymised

Telegram text may be laid over media as chat bubbles with typing animation.
**No author names.** Two bubble sides (left/right) distinguish the two voices
without naming them. This is understood by the audience as a chat export.

### 1.8 Upscale at conform, never at draft

Delivery is 1080p (4K optional, `deliver.resolution`). Sources below delivery
resolution — Telegram media, and any 360 recording where only the `.lrv`
exists — are upscaled with a learned upscaler on the GPU, capped at delivery
resolution, and sharpened no further than `deliver.max_upscale_factor` (2.0)
allows; beyond that the frame is padded rather than invented. Phone originals
at 1080p/4K are conformed, not upscaled.

---

## 2. Architecture changes

### 2.1 Stage map (what moves)

```
S01 Probe            unchanged + prune step + ignore list
S02 Spine            + Strava track, + Act 6 boundary, + hallucination filter; music analysis unchanged
S06 Assemble v2      … then music chosen per scene from its attributes (§5.6)
S03 Process          unchanged; bugs fixed (has_face, video geotag, day_index)
S04 Semantic         S04.1 CLIP (remote GPU) · S04.2 framing (Claude, 2x2 sheet) · S04.3 captions (Claude)
S04.5 Beat sheet     NEW: Claude reads transcripts + chat → beats.json           [remote, API]
S05 Score            unchanged formula; terms now actually populated
S06 Assemble v2      beats first, then picture fill; rhythm from music; constraints live
S06.5 Overlays       NEW: HUD frames, cards, map, profile, credits — PNG/MOV assets [remote CPU]
S07 Draft render v2  two-track audio graph, motion filters, overlays          [remote CPU]
S08 Conform          originals, proper stitch, upscaling                      [remote GPU]
S09 Deliver          both cuts + credits
```

Every new stage is a `stage_units` unit, resumable, and pure where it decides.

### 2.2 Execution model: local decides, remote works

**The local machine computes nothing** (operator, 2026-09-18: "my machine is
dying and very slow — let's move to GCP completely"). It edits files, runs
git, and issues `nepal remote` commands. Scoring, assembly, tests, probes,
renders and API loops all run on the remote box:

```bash
nepal remote up                       # create or resume the box; it pulls the bucket
nepal remote run s04 --redo clip      # run a stage there, stream the log, push results to the bucket
nepal remote exec -- pytest -m slow   # any command on the box
nepal remote pull                     # bucket -> local, for inspection
nepal remote down                     # stop (keep the disk) or --delete
```

**The bucket is the hub.** `gs://nepalvideo-29922345852` (europe-west4) holds
`raw/` (phones, chat, music, Strava; not the camera originals until conform),
`work/` and `ref/` (SRTM, GeoNames). Local pushes to it, the box pulls from it
before a run and pushes after, local pulls to look. The database is
authoritative in the bucket from the first push on; a VM is disposable.

Provider: **GCP**, project `nepalvideo`. `e2-standard-8` Spot (~$0.10/h) for
CPU stages and API loops; `g2-standard-4` with one L4, Spot (~$0.30/h), for
CLIP, faces and upscaling once the global GPU quota lands. Both stop rather
than delete on `down`, so the disk (~$5/month) is the only idle cost. A
serverless GPU remains the fallback for GPU minutes if the quota stalls.

Claude calls go **directly to the Anthropic API** from the remote box
(`ANTHROPIC_API_KEY` in the box's environment, never in the repo), using the
Batch API wherever the stage is not interactive. Estimated spend for the whole
v2 run, at published rates:

| Item | Estimate |
|---|---|
| Beat sheet (one long-context call, text only, Sonnet 5) | < $1 |
| S04.2 framing, 22 dual-fisheye recordings' shots, Haiku 4.5 | < $1 |
| S04.3 captions, 1,632 shots × 4 frames, Haiku 4.5 (Batch) | ~$1.5 |
| Remote CPU, ~8 h over the project (includes the Blender flyover) | ~$1 |
| Remote GPU, ~1 h total (CLIP minutes, upscaling at conform) | ~$0.5 |
| **Total** | **~$5, ceiling $15** |

Every paid stage logs `usage` to `work/reports/` and refuses to start if the
running total in `decisions('spend_usd')` would exceed `cloud.spend_ceiling_usd`
(15).

What travels: nothing before S08 reads the originals. `nepal remote up` syncs
proxies, audio, transcripts, faces, the database and the stills (~9 GB) once,
then only the database and new artefacts. S08 syncs the originals of the
selected recordings only (~120 files).

---

## 3. Data model changes

### 3.1 `shots` (fixes, no new semantics)

- `has_face` set from `face_score > 0` at recluster time as well as at
  detection, so re-detecting shots cannot leave it at the default.
- `lat`, `lon`, `alt_dem_m`, `place_name`, `day_index` populated for video
  shots by S03.2 (interpolate → DEM → gazetteer, same functions S03.0 uses for
  stills). `day_index` is the trek day (Act 2 day 1 = 1); planning and after
  material carry NULL.
- `transcript_json` (new, TEXT): whisper segments `[{start, end, text}]` on the
  recording's clock, so a beat can be cut at an utterance. Today they exist
  only as files under `work/transcripts/`.
- `hallucinated` (new, INTEGER DEFAULT 0): set by the S02/S03 filter (§4.1);
  a hallucinated shot has `has_speech = 0` for every downstream purpose.

### 3.2 `beats` (new)

```sql
CREATE TABLE beats (
  beat_id     TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,      -- speech | quote | closing | title
  act         INTEGER,
  shot_id     TEXT REFERENCES shots(shot_id),   -- speech: the recording it comes from
  msg_id      TEXT REFERENCES messages(msg_id), -- quote/closing: the message
  src_in      REAL, src_out REAL,               -- on the recording's clock
  text        TEXT,                              -- what is said or shown
  levity      INTEGER DEFAULT 0,
  effect      TEXT,                              -- optional: freeze | burst | ramp | none
  rationale   TEXT,                              -- why Claude picked it, for Gate 2
  rank        INTEGER                            -- Claude's order of importance
);
```

### 3.3 `timeline` becomes picture slots; `audio_cues` and `overlays` are new

```sql
-- picture track
CREATE TABLE timeline (
  slot_index  INTEGER PRIMARY KEY,
  act         INTEGER,
  t_in REAL, t_out REAL,
  kind        TEXT NOT NULL,      -- video | photo | card | title | map | credits | black
  shot_id     TEXT REFERENCES shots(shot_id),
  src_in REAL, src_out REAL,
  secondary_shot_id TEXT,         -- split screen: the other phone's clip
  secondary_src_in REAL,
  motion      TEXT,               -- JSON: {"type":"yaw_drift","from":..,"to":..} | {"type":"ken_burns",...} | {"type":"crop_face",...} | {"type":"blur_fill"}
  speed       REAL DEFAULT 1.0,   -- speed ramp; 1.0 is real time
  transition  TEXT,               -- cut | dissolve | dip_black
  beat_id     TEXT REFERENCES beats(beat_id)     -- the beat this picture serves, if any
);

-- audio track(s)
CREATE TABLE audio_cues (
  cue_id      TEXT PRIMARY KEY,
  track       TEXT NOT NULL,      -- speech | location | music
  t_in REAL, t_out REAL,
  source      TEXT,               -- recording_id or track_id
  src_in REAL, src_out REAL,
  gain_lufs   REAL,               -- target integrated loudness for this cue
  fade_in_s REAL DEFAULT 0, fade_out_s REAL DEFAULT 0,
  beat_id     TEXT REFERENCES beats(beat_id)
);

-- generated pictures laid over the picture track
CREATE TABLE overlays (
  overlay_id  TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,      -- hud_map | hud_day | place_card | profile | chat_card | stat_card | subtitle
  t_in REAL, t_out REAL,
  payload     TEXT,               -- JSON the renderer needs (text, lat/lon, altitude, day, bubble side …)
  asset_path  TEXT                -- rendered PNG/MOV under work/overlays/, filled by S06.5
);
```

`timeline.otio` and `.fcpxml` are still written: picture slots as one video
track, `audio_cues` as three audio tracks, overlays as a fourth video track
referencing the rendered PNG/MOV assets. The cut stays refinable by hand.

---

## 4. The voice spine

### 4.1 Hallucination filter (deterministic, before anything reads a transcript)

A transcript segment is marked hallucinated when any of:

- its text matches `asr.hallucination_phrases` (config; seeded with "Субтитры
  сделал", "DimaTorzok", "Продолжение следует", "Редактор субтитров", "Спасибо
  за просмотр", "Подписывайтесь");
- the same 3-gram repeats ≥ 3 times inside the segment;
- whisper's `no_speech_prob` > `asr.max_no_speech_prob` (0.6) or
  `compression_ratio` > `asr.max_compression_ratio` (2.4) — both are already in
  the JSON faster-whisper writes;
- the segment lies inside a window where the VAD measured < `process.speech_min_s`
  of voice.

A shot whose every segment is hallucinated gets `hallucinated = 1`. Re-running
the filter is seconds; the transcripts are not re-made. Applied in S03 next to
the gate, and reported.

### 4.2 The beat-sheet call (S04.5)

**Input** (one request, text only, ~60–80k tokens):

- every non-hallucinated transcript with segment times, tagged with recording,
  day, place, altitude, source, who is on screen (face cluster) and whether the
  picture is a piece-to-camera (face fills the frame) or a walking shot;
- the Telegram thread, both phases, with timestamps and an anonymous speaker
  tag (A/B);
- the day table (`nepal report`): date, place, altitude, km, gain, shots,
  photos, messages;
- the six-act structure and the brief (§1 of the spec, verbatim);
- the vocabulary (`work/vocab/`).

**Output** — strict JSON validated against a schema, retried once on failure:

```json
{
  "title": "…",
  "beats": [
    {"kind": "speech", "shot_id": "camera_…#0003", "src_in": 41.2, "src_out": 52.8,
     "act": 3, "text": "Я вот на этом курумнике прям сдох…", "levity": true,
     "effect": "none", "rank": 2, "rationale": "the first admission of cost"},
    {"kind": "quote", "msg_id": "…", "act": 1, "text": "…", "rank": 7, "rationale": "…"}
  ],
  "pairs": [
    {"planning_msg_id": "…", "trek_beat_id": "…", "why": "20 km a day, typed in February"}
  ],
  "closing": {"msg_id": "…", "text": "…"},
  "act_notes": {"3": "the climb should feel slower after the bridge…"},
  "stat_card_ideas": ["…"],
  "trailer": {"hook_beat_id": "…", "cliffhanger_beat_id": "…", "cards": ["…"]}
}
```

`pairs` are the planning-versus-reality moments (§13.2): a confident message
from the planning phase laid as a chat card over the trek beat that answers
it. Zero to five; the model returns none rather than forcing one.

**Rules the prompt states and the validator enforces:**

- 10–16 speech beats, chronological, ≥ 1 per act from Act 2 on, ≥ 1 levity per
  act from Act 2 on, none longer than `beats.max_speech_s` (25) — a beat may
  join adjacent segments of one recording;
- `src_in`/`src_out` must lie on segment boundaries that exist in the input;
- 4–8 Act 1 quotes and 1 closing line, each ≤ `spine.card_max_chars`;
- Act 4 gets at most one beat, and the summit words ("Покорена", the pass
  altitude) are candidates the prompt names explicitly;
- the model is told what it must **not** do: invent text, choose hallucinated
  segments, or exceed the count.

The result is written to `beats` and `work/beats.json`, and shown at **Gate 2**
with the shortlist: the operator can drop, re-rank or retime a beat. The
beat sheet is re-run only on `--force`; it costs a dollar and the answer should
be stable.

### 4.3 What happens to `speech_first` and `has_speech` weight

`speech_first` is removed. `score.ctx.has_speech` stays but applies only to
non-hallucinated speech and is no longer a claim on a slot: the beats are.

---

## 5. Assembly v2

Pure functions in `process/assemble.py`, driven by `stages/s05_cut.py`. Order of
operations per act:

### 5.1 Anchors first

Each beat becomes an **audio cue** on the `speech` track at the position the
act's chronology puts it, with its full utterance. Under it, picture slots:

- the first `beats.face_hold_s` (2.5) seconds show the speaker's own shot when
  the beat's recording is a piece-to-camera (`face_score` high, face fills the
  frame), cut at the utterance's `src_in` minus `beats.pre_roll_s` (0.4);
- the remainder is **B-roll**: picture slots chosen by MMR from shots within
  ±`beats.broll_window_s` (7,200) of the beat's timestamp, same act, excluding
  the beat's recording — the voice continues (an L-cut);
- if the beat is a walking shot rather than a piece-to-camera, the picture stays
  on the beat's own recording for the whole utterance (people talking while
  walking is the trek).

Quote beats become `chat_card` overlays over the picture fill of Act 1 (and,
sparingly, elsewhere when Claude placed one); the closing beat becomes the
final card before credits.

### 5.2 Picture fill between anchors

The gaps between anchors are filled chronologically. Candidates are the
shortlisted shots whose timestamp falls in the gap's window. Selection is MMR
(`score_total − λ·max cosine`) with CLIP embeddings; **without embeddings the
fallback similarity is deterministic**, not zero: 1.0 for the same recording
within 60 s, 0.6 for the same recording, 0.3 for the same place and hour. That
alone removes the runs of one recording the first draft shows.

Constraints, applied as admissibility filters at every step (all exist in code
today; now they are called):

| Constraint | Value | Where |
|---|---|---|
| stills share per act | `film.photo_share` 0.10, via `select.plan_photo_slots` | never two stills adjacent unless in a burst |
| subject | ≥ 1 shot with a face per `assemble.subject_shot_every_s` (40) | `needs_subject` |
| levity | ≥ 1 per act from Act 2, from beats or `tag_levity` | `missing_levity` |
| place | ≤ 3 per place per act | `place_count_ok` |
| recording | ≤ 2 consecutive slots from one recording outside a beat | new |
| source alternation | after 3 consecutive slots from one source, prefer another | new, soft |
| chronology | non-decreasing within act; relax diversity before chronology | existing |

### 5.3 Two phones, one moment

Before the fill, S06 finds **pairs**: a `phone_keller` and a `phone_kulikov`
clip (or a phone clip and a camera shot) whose corrected timestamps overlap
within `assemble.pair_window_s` (60), both portrait or one portrait. A pair is
one slot of `kind = video` with `secondary_shot_id` set and
`motion = {"type":"split"}`; both halves run in sync from the same instant.
At most `assemble.pairs_per_act` (2). Pairs are ranked by the sum of scores
and by whether both faces are different people — the point is to see both of
them at once.

### 5.4 Rhythm from the music

`music_sections.energy` is percentile-ranked within the act's cue. A slot's
target length comes from the section under its start:

| section energy percentile | target length | cut point |
|---|---|---|
| < 33 | 5–8 s | nearest beat |
| 33–66 | 3–5 s | nearest beat |
| > 66 | 1.5–2.5 s | nearest downbeat |
| the act's highest swell | **burst**: 8–12 slots of 1 beat each | every downbeat |

Beats' audio is never cut by this; only picture under it changes at the rhythm
the music asks. Act 4's peak: the burst lands on the swell, then one held shot
(`act4_held_shot_s`, 6–10 s) as the music cuts to the 5 s silence window with
location sound at full.

Natural-sound windows (`assemble.natural_sound_windows`, 4): chosen by
`effort.hardest_windows` from the GPS profile; music fades out over 1 s,
location audio at full, picture stays on the recording that has the sound.

### 5.5 Act 0 and credits

Act 0 is the opening of §13.8 (the rise and the flyover, title landing on the
pass) followed by the cold open: the highest-ranked Act 3/4 beat or, absent
one, the highest-scoring Act 4 shot; 15–25 s; music is the Act 4 cue from its
swell; hard cut to black; the card "three months earlier" (config
`film.cold_open_card`), then Act 1.

Credits per spec §1.8 and §S09.1, generated from the database at render time;
`decisions.credits_track` under them. Rendered with the draft so Gate 3 sees
them.

### 5.6 Music chosen by the scene

The library is pre-defined (the tracks in `music/`), nobody is asked about it
at a gate, and no track title is shown during the film. What changes is
*how* a piece gets chosen: not one track per act from five averaged features,
but a section of a track per **scene**, from what the scene is.

**Scenes.** After the picture fill, consecutive slots are grouped into scenes
while they share an act and an activity class and their speed and heart rate
stay within one band. A scene is at least `music.min_scene_s` (45) and at
most `music.max_segment_s` (150) long, so the music neither flickers nor
swallows an act. Scene boundaries are where the music may change; a change
never lands inside a beat's audio.

**Scene attributes**, all already in the database or derivable from it:

| Attribute | Source |
|---|---|
| part of the trek | act, `day_index`, the Strava stage name |
| speed | Strava fix at the slot's time, else GPS-derived (`effort.profile`) |
| effort | heart rate (§13.1) blended with gain rate and slowness |
| activity class | `walking · climbing · descending · resting · crossing · village · summit · city · transport`, from speed, gain sign, altitude band, place name, act and caption keywords |
| altitude and time of day | `alt_dem_m`, `effort.light_quality` |
| voice | share of the scene under speech beats |
| levity | any `tag_levity` or beat-sheet levity inside the scene |
| picture energy | mean `motion_mag`, the rhythm class of §5.4 |

**Track attributes**, from S02.7's analysis, per section rather than per
track: energy (percentile within the library), tempo, onset rate, spectral
centroid, dynamic range, and the section's role in its piece (intro, build,
swell, outro) from `is_swell` and the energy trend. `music.tag_with_llm`
(off) lets Claude add mood tags per track from title and artist for a few
cents; the matcher works without them.

**Targets.** A pure function `scene_target(scene) → {energy, tempo,
dynamics, brightness}` in `spine/music.py`, weights in `music.scene_targets`:
energy rises with effort and with the act's place in the climb and falls in
villages, at rest and in the city; tempo follows the walking cadence (Strava
cadence where present, else from speed), accepting half and double time;
dynamics are wide for climbing and narrow under speech; brightness follows
time of day and altitude (pre-dawn and the pass are dark, villages and
Kathmandu bright). Act 1 and Act 6 bias toward the same piece so the
callback happens by construction.

**Assignment.** A Viterbi pass over the scenes with track sections as
states. The cost of a (scene, section) pair is the weighted distance between
the scene's target and the section's features, plus a switching cost when
the section's piece differs from the previous scene's, minus a continuity
bonus when it is the section that follows in the same piece, plus a
repetition penalty for a piece heard within `music.reuse_gap_s` (300).
Hard rules: **Marusha's *Somewhere over the Rainbow* is the credits piece
and plays there only** — it is withdrawn from the library before the scene
assignment runs, as `music.credits_track` already does, so no scene can pick
it and the credits are the first time it is heard; Act 4's last scene must
end on a swell, followed by the silence window; the long take (§13.5) and the
natural-sound windows carry no music; Act 0 takes the section chosen for
Act 4 from its swell so the summit music is heard first and recognised when
it returns. The result is written to `music_map.json` in the existing shape
(segments with `src_in`/`src_out` per act) so §5.4 and the audio graph read
it unchanged; `assignment_note` records the per-scene reasoning for the run
report, never for the screen.

The per-act Hungarian solve stays available as `music.assignment: act` for
comparison; `scene` is the default.

---

## 6. Picture treatment and overlays

### 6.1 Motion for every slot (`timeline.motion`)

| Source | Default motion | Notes |
|---|---|---|
| still | `ken_burns`: 1.00→1.08 zoom, pan direction alternating per slot, eased | `render.ken_burns_zoom` |
| portrait video, no face | `blur_fill`: the clip centred over its own blurred, scaled copy | standard vertical-in-horizontal |
| portrait video, face | `crop_face`: 16:9 crop tracking the face box from S03.6 samples, smoothed | falls back to `blur_fill` if the face leaves |
| 360 video | `yaw_drift`: rectilinear 100°×70° view from the equirect, yaw moving 8–15° over the slot toward the chosen yaw | chosen yaw from S04.2, else `face_yaw`, else the yaw with the highest Laplacian variance among four |
| 360, summit reveal | `tiny_planet` once, on the Act 4 held shot | `render.tiny_planet: true` |
| walking shot in Act 3, no speech | `speed = 2.0` ramp when the slot is > 4 s | `render.ramp_acts: [3]` |
| long static camera runs (> 3 min, low motion) | `speed = 30` timelapse slot | candidates from `motion_mag` p10 |

Effects are a vocabulary the assembler chooses from with caps per act
(`render.effects_per_act`: 3), plus what the beat sheet suggested per beat. A
freeze-frame with a stat card is the one effect allowed to stop time: at most
once per act.

### 6.2 Where the yaw comes from (S04.2 built cheaply)

For each shortlisted 360 shot, a 2×2 contact sheet of the four yaw views goes
to Claude (Haiku 4.5) with the question the spec asks: best composition, and
whether a person is the subject. Fewer than 200 shots; under a dollar. Where
`has_face = 1`, the subject view competes as a second candidate, as the spec
says.

### 6.3 Draft versus conform

The draft renders from the proxies at 960×540, 30 fps, with every motion and
overlay in place, so Gate 3 judges the real film small. Conform (S08) applies
the same `timeline`, `audio_cues` and `overlays` to the originals: proper
stitching for 360 shots where a stitcher exists, `v360` at full resolution
where not, upscaling per §1.8, one LUT per source class.

### 6.4 Overlays (S06.5, generated; S07 composites)

All overlays are rendered to PNG sequences or ProRes-alpha MOVs under
`work/overlays/` by a pure layout function plus PIL/matplotlib, on the remote
CPU box, then composited by ffmpeg `overlay`. Nothing is drawn with `drawtext`.

| Overlay | When | Content |
|---|---|---|
| `hud_map` (bottom-left, ~180 px) | whole trek, Acts 2–5 | route from the GPX on a muted OSM basemap (tiles fetched once by `nepal fetch-reference`), the walked part filling in, a dot at the current position, current altitude in a monospace readout |
| `hud_day` (top-right) | whole trek | `Day 6` — from `day_index`; Acts 1 and 6 show the date instead |
| `place_card` | first slot at a new `place_name` | `Day 6 · Syalagaun · 4,068 m`, 3 s, lower third |
| `profile` | each act transition, 3 s | the elevation profile, filled to the current day, the pass marked |
| `chat_card` | Act 1 quotes, sparingly elsewhere | chat bubble, left/right by speaker, typing animation, timestamp, **no name** |
| `stat_card` | once per act, on a freeze or a hold | generated from the DB: km, gain, hours walked, highest point, photos taken, messages sent, coldest morning |
| `subtitle` | under every speech beat | the transcript text, bottom centre, since the audience may not be Russian speakers — `render.subtitles: true` |
| `name_tag` | first appearance of each named face cluster | an arrow-style tag from the face box: `Darma · guide`, and the two protagonists by first name (`overlays.name_tags`, §13.7) |
| `peak_label` | when Manaslu is in frame | arrow to the summit's pixel and `Manaslu · 8,163 m · 14 km` (§13.3) |
| `hud_hr` (top-left, small) | Acts 3–5 where Strava has heart rate | a pulse at the real rate with the number (§13.1) |
| `credits` | after the last slot | the four blocks of spec §1.8, plus the disclosures of §13.10 |

Anything a widget shows is computed from `gps_points`, `shots`, `messages` and
`assets` at layout time; nothing is typed.

---

## 7. Audio graph (S07/S08)

One ffmpeg filter graph, built as data:

- **speech track**: each `audio_cues.track = 'speech'` cue from the recording's
  extracted `.wav` (draft) or original (conform), `loudnorm` to
  `render.speech_lufs` (−16), fades of 0.15 s;
- **location track**: every picture slot's own audio at `render.duck_lufs`
  (−28) under music, at `render.location_full_lufs` (−18) inside natural-sound
  windows and the silence window, at −24 under speech; a still or a card
  contributes the previous slot's ambience, held;
- **music track**: the act's segments with `render.music_xfade_s` (2.0)
  crossfades, at −14, ducked to −22 under speech cues by an explicit volume
  envelope from the cues (no sidechain — the timing is known), faded out over
  1 s into windows and the Act 4 silence, dead for the 5 s;
- mixed with `amix` (normalize=0), final `loudnorm` to −14 LUFS, true peak
  −1.5 dB; encoded AAC 192k.

`process/mix.py` becomes the pure function that turns `audio_cues` plus
windows into the per-track volume envelopes; it stops being dead code because
`render.build_command` consumes it.

---

## 8. Bugs and hygiene folded into this work

Each is a unit test plus, where a tool boundary is touched, a slow test.

1. **Prune step** (`nepal prune`, also run by S01): delete `recordings`,
   `shots`, proxies, audio and transcripts for anything the manifest no longer
   produces; delete recordings whose grouping method the current code would
   not produce (the `phone_*_IMG` ghosts and their 230 MB proxy). Report what
   went.
2. **Manifest ignore list** (`probe.ignore_globs`: `*.zip`, `aws/**`, `*_thumb.jpg`).
3. **`has_face`** derived from `face_score` at gate time, like stability.
4. **Video shot position**: lat/lon/alt/place/day for video shots in S03.2;
   `nepal diagnose` reports positioned counts per kind so a regression is
   visible.
5. **`day_index`** computed from the act-2 start for every shot and asset.
6. **Whisper segment times** stored (`transcript_json`) and used for beats.
7. **Hallucination filter** (§4.1).
8. **Output frame rate** fixed at `render.fps` (30); every input `fps=`-filtered.
9. **Photo budget** and the four constraints actually invoked (§5.2).
10. **FOV re-solve** on the 22 dual-fisheye recordings (`nepal s01 --redo fov`
    is added so it does not need `--force`); `fov-check` sheets at Gate 1.
11. **The 3 undated `phone_kulikov` clips**: a `decisions` override
    `capture_time_overrides` the operator can fill; otherwise they stay out.
12. **`usable_as_card`** is dropped as a selector (it marks 79% of messages);
    quotes come from the beat sheet.
13. **Strava ingestion** (§13.1): `nepal_data/strava/activities.csv` plus the
    `.fit.gz` files become `gps_points` rows with `source = 'strava'` and the
    new `hr_bpm`, `alt_baro_m` columns; `spine/gps.py` merges them ahead of
    photo fixes. Pure-Python FIT decoding (`fitdecode`) is a core dependency.
14. **Heading and lens tags** added to `CAPTURE_TAGS` (`GPSImgDirection`,
    `GPSImgDirectionRef`, `GPSHPositioningError`, `FocalLengthIn35mmFormat`)
    and stored on `assets` as `heading_deg`, `pos_error_m`, `focal_35mm`; the
    photo manifest is re-probed once (minutes, on the remote box).
15. **Named peaks** in `reference.py`: Manaslu (28.5497 N, 84.5597 E, 8,163 m)
    and the other 7,000 m+ peaks of the circuit from GeoNames, in
    `spine.named_peaks`, so the geometry of §13.3 has targets.

---

## 9. Gates (what the operator sees)

- **Gate 1** gains: the six act boundaries with the Strava stage names; the
  FOV sheets; the Delhi footage check ("Act 6 has N minutes of material");
  Strava coverage per day (which days have heart rate and a 1 Hz track).
  Music is not on the page.
- **Gate 2** gains: the beat sheet (text, rationale, play button on the
  utterance), drop/retime/re-rank; the pairs; the 360 framing choices.
- **Gate 3**: the draft with everything in it, credits included, and the shot
  list keyed by wall-clock time and slot index (the burned overlay is replaced
  by a generated `subtitle`-style slot label in draft mode only).

Gates stay static pages written under `work/gates/` and opened locally; the
Step Functions machinery remains out of scope.

---

## 10. Testing

- Pure: beat validation against the schema and the rules; MMR fallback
  similarity; rhythm mapping from section energy; pair finding; envelope
  generation from cues; overlay layout (payloads, positions, no name in any
  chat card).
- Slow, real tools: a two-track render of a synthetic timeline with speech,
  music and a silence window, checked by `ebur128` per window; a `yaw_drift`
  render of the fixture dual-fisheye clip; `crop_face` on a synthetic portrait
  clip with a moving rectangle; an overlay composite with alpha; the prune step
  on a fixture tree from which a file was removed.
- Contract: the Claude calls are behind thin wrappers with recorded responses
  in tests; one opt-in live test (`NEPAL_LIVE_API=1`) that costs cents.

---

## 11. Order of work

1. Bugs and hygiene (§8), including Strava ingestion and the heading tags —
   independent of every creative decision; re-cut and re-render the cheap
   parts to confirm the stills/videos balance and the ghost removal.
2. Remote execution (`nepal remote`) and the spend guard — everything after
   this runs there.
3. Hallucination filter, segment times, beat sheet (S04.5) with the pairs and
   the trailer picks → Gate 2 review of the beats.
4. Schema v2 (`beats`, `timeline`, `audio_cues`, `overlays`) and assembly v2
   (§5) with the deterministic similarity fallback, the long take (§13.5);
   audio graph (§7).
5. CLIP on the remote GPU; S04.2 and S04.3 via the API, the latter asking
   about Manaslu (§13.3).
6. Motion (§6.1), grade by altitude (§13.4), overlays (§6.4) including the
   name tags, the peak label and the cards of §13.6; the mountain-approaches
   motif (§13.3); credits; the opening (§13.8) and the cold open.
7. Six acts and the Gate 1 page.
8. Conform with upscaling (§1.8) once the Delhi footage is in and Gate 3 has
   passed; then the vertical option and the trailer (§13.9).

Each step ends with a draft the operator can watch.

---

## 12. Open items for the operator

- **Delhi footage**: add it under `nepal_data/` (any folder; the manifest walks
  the tree). Say when it is there; the prune-then-resume path handles the rest.
- **Provider**: GCP account and a GPU quota request, or a serverless-GPU
  account, or an SSH box you already have. Credentials stay with you.
- **Subtitles**: on by default; say if the audience is Russian-only.
- **The porters' names** for the credits, if you want them credited by name.
- **Name tags** (§13.7): default is all three people on first appearance;
  say if only Darma should get one.
- **Bridge foley** (§13.5): off by default; turn on `render.long_take_foley`
  if the real audio of the crossing is not enough.

---

## 13. Additions adopted from the 2026-09-16 brainstorm

### 13.1 Strava is the spine

`nepal_data/strava/` holds `activities.csv` and eleven `.fit.gz` files, one
per trekking day from *Machakhola to Jagat* (29 April) to *Bimtang to
Dharapani* (7 May), recorded on an Apple Watch: position at roughly one fix
per second, barometric altitude, heart rate (peaks of 126–160 per day), and
per-activity distance, moving time and elevation gain. This replaces photo
EXIF as the primary track wherever it exists; photos fill the jeep days and
Kathmandu.

What changes because of it:

- `gps_points` gains `hr_bpm`, `alt_baro_m`, `activity_id`; S02.1 reads Strava
  first and photo fixes second; S02.3 prefers barometric altitude to SRTM
  where both exist and records which was used.
- `spine/effort.py` gets a heart-rate term: exertion is `hr / hr_max_observed`
  where heart rate exists, blended with the GPS-derived terms; the
  natural-sound windows and the Act 3 burst follow it.
- Days are named by the activity (`Day 3 · Deng to Namrung`) and bounded by its
  start and end; the pre-dawn summit start is the activity's start time
  (*Dharmasala to Larkya Pass* began at 04:07 local).
- Stat cards use the activity's own numbers (distance, moving time, gain,
  maximum heart rate) rather than derived ones.
- A small `hud_hr` widget shows the live rate through Acts 3–5; it hides
  wherever there is no data rather than freezing.
- Watch time is GPS-disciplined, so the clock solver gains a third reference
  to check the phones against; a disagreement above 2 s is reported.

### 13.2 Planning versus reality

The beat sheet returns up to five `pairs`: a confident planning message and
the trek beat it collides with. Rendered as a `chat_card` (anonymous, dated)
sliding in over the trek picture while the beat's audio plays. The model may
return none; a forced pair is worse than no pair.

### 13.3 Manaslu, tracked and labelled

Two mechanisms, geometry first:

- **Geometry, for photographs.** iPhone stills carry `GPSImgDirection` (true
  north) and a 35 mm-equivalent focal length. For each positioned photo with a
  heading: the bearing and elevation angle from the camera to the summit are
  computed from the two positions and altitudes; the summit is in frame when
  the bearing lies inside the horizontal field of view and the elevation
  angle inside the vertical one; a line-of-sight test along the bearing over
  the SRTM tiles rejects a summit hidden behind a nearer ridge. The result is
  the summit's pixel position, within `overlays.peak_tolerance_deg` (3).
- **Recognition, for video.** S04.3 captioning asks, for every shot, whether
  Manaslu (the double-summited peak) is visible and roughly where; a yes with
  no geometry places a corner `peak_label` without an arrow.

The **mountain approaches** motif: photographs where geometry found the
summit, one per day of the approach, cropped so the summit sits at the same
screen position in each, played as a 6–8 s sequence on the beat, the mountain
growing in place with the day counter ticking. Placed once, in Act 2 or early
Act 3, where the beat sheet's act notes suggest.

### 13.4 Grade by altitude

The colour temperature of the grade follows the shot's altitude: linear from
`render.grade.warm_k` (5,600 K) at `render.grade.low_m` (800 m) to
`render.grade.cold_k` (7,400 K) at `render.grade.high_m` (5,200 m), applied per
slot with ffmpeg's `colortemperature` filter in the draft and folded into the
per-source LUT at conform. Acts 1 and 6 sit at the warm end. Subtle by design;
the audience should feel it, not see it.

### 13.5 The long take: the bridge

One slot of `assemble.long_take_s` (60–90 s), unbroken, on the highest
suspension-bridge crossing the corpus holds. Candidates: recordings whose
captions mention a bridge or whose transcripts do ("мост"), ranked by drop
below the bridge from the DEM at the crossing point and by the recording's
continuous length. Picture is the recording as shot, `yaw_drift` off, one
slow pan at most. Audio: music fades out fully 2 s before, the recording's
own sound at full for the whole take (creaks, wind, boots on the deck), music
returns on the far bank. `render.long_take_foley` (off) allows a licensed
sound layer on top if the real audio is wind-blown; when it is on, the
credits say "sound design" in the machine block.

### 13.6 Oxygen and sunrise cards

Two `stat_card` variants computed from altitude, position and time alone:

- **Oxygen**: the barometric formula gives the pressure at the altitude and
  hence the share of sea-level oxygen; shown on the first slot above 4,000 m
  and at the pass: `5,106 m · 53 % of the oxygen at sea level`.
- **Sunrise**: solar position for the date and place gives sunrise and
  golden hour; shown on the summit day with the actual start time from
  Strava: `Left Dharmasala 04:07 · sunrise 05:19`.

### 13.7 Name tags

On the first appearance of each named face cluster, a `name_tag` overlay:
a thin line from the face box to a label for `overlays.name_tag_s` (2.5 s).
Labels come from `overlays.name_tags` (`keller: Kirill`, `kulikov: Sasha`,
`other_0: Darma · guide`); the cluster-to-person mapping is confirmed at
Gate 2 as the spec already requires. Chat cards remain anonymous; the tag
names people on screen, which is a different promise.

### 13.8 The opening: the rise and the flyover

Two pictures, dissolved into one move, before the cold open:

1. **The rise** (8 s). A 360 frame with both protagonists in it, ideally the
   trailhead or a viewpoint early in Act 2 (chosen by face count and
   `vlm_interest`, overridable in `decisions`). The reprojection animates
   pitch from level to straight down and the field of view from 100° to a
   full tiny planet: the camera appears to lift away and look down at the two
   figures. This is real footage and needs no drone.
2. **The flyover** (12 s). A terrain render from the SRTM tiles: hillshade
   with hypsometric tint, snow above 5,000 m, the route drawn progressively,
   labels rising at the key places (Machakhola, Jagat, Deng, Namrung,
   Samagaon, Manaslu 8,163 m, Larke Pass 5,106 m, Bimtang, Dharapani). The
   camera follows the route from the trailhead to the pass and the title
   lands there. Rendered headless in Blender (`bpy`, Eevee) on the remote CPU
   box from a generated script; a 2.5-D `pyvista` render is the fallback if
   Blender is not installable. The same scene, reversed, closes Act 6 before
   the credits, and short cuts of it serve as the act-transition map moments.

The flyover is the one picture in the film not shot on the trek; the credits
say so (§13.10).

### 13.9 Formats: horizontal film, vertical option, vertical trailer

- The film is 16:9. `deliver.vertical_video: false` by default; when true, a
  9:16 version is rendered from the same timeline: portrait clips native,
  landscape shots cropped to the face or to saliency, 360 shots reframed at a
  9:16 field of view, HUD widgets stacked at the top, cards resized.
- The short cut becomes a **vertical trailer**, 60–90 s, `deliver.trailer:
  true`: hook (3 s of the peak), three planning quotes as cards, a burst
  montage on the Act 3 build, the cliffhanger — the pre-dawn start, breathing,
  "Страшно", cut to black on the hardest moment before the pass — then the
  title. It never shows the pass. The beat sheet's `trailer` block names the
  hook and the cliffhanger; the assembler's `trailer` profile builds it with
  the same code paths and the burst rhythm of §5.4.

### 13.10 What the credits disclose

The machine block of the credits states, generated from the run: every frame
except the flyover map was shot on the trek; how many shots were considered
and used; that the transcriber hallucinated "Subtitles by DimaTorzok" 33
times; that the camera's clock was 18 days wrong; whether sound design was
added to the bridge; and that the Nepal painting in the chat was imagined
before anyone went.
