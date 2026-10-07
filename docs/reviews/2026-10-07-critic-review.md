# Critic's review — draft of 2026-10-07 05:43 (25.9 min, 440 slots)

Seen as 680 frames: one from the middle of every slot and three (in, mid,
out) from every slot of four seconds or more, tiled into 34 contact sheets
in film order; the timeline, cue and overlay tables; loudness per spoken
line and per music window (ffmpeg ebur128). Written as an outside reviewer
would, against the film the pipeline has made so far, not against the
pipeline. Rulings the operator has already made (Gate 3, rounds one to
three) are taken as given and not re-argued.

## What works

- **The spine.** Sixteen spoken lines carry the story from the kitchen
  table to the pass and back; each now sits on its own picture for its
  whole length. The callback of the opening line at 20:51, delivered where
  it was said, is the best structural idea in the film.
- **The pass.** 20:04–21:55 is a real climax: the pre-dawn start (#309–
  #311), the headlamp selfie (#313), the sign (#325–#345), the group at
  the top. It earns its swell.
- **Sound at the cuts.** Picture and sound are now on the same clock
  (within one frame end to end); the room tone continues under stills;
  music enters on a cut four seconds after a line and never cuts to another
  point in the same track inside a window.
- **Music placement** in this cut, the operator's own words: keep it.
  Eight windows, Acts 2 to 5, entering on movement after talk.
- **Chronology** is strict within every act, and the two phones' takes of
  one scene are both allowed, which gives the river crossings (#66–#67) and
  the bridges (#57–#60, #162–#166) their best moments.

## What hurts

Ranked by how much of the film it touches. Timestamps are film time;
`#n` is the slot.

1. **A third of the picture is a thin strip between black bars.** Every
   portrait phone clip (kulikov 720×960, keller 1080×1920) and every
   Telegram round video is letterboxed into a 16:9 frame that is mostly
   black: sheets 1, 14, 16–19, 24, 31–34 are more black than picture.
   Act 1 (0:18–1:13) is entirely circles on black. The cremation sequence
   at 24:39–25:05 is fourteen portrait slots of one clip. This is the
   single largest visual cost in the film.
2. **Repetition of the same recording.** The run rule (`assemble.
   max_consecutive_recording`) does not hold through the refill rounds and
   the chronological re-sort: 30 runs of three or more consecutive slots
   from one recording, the longest twelve (the jeep, 2:14–3:01, 47 s of
   the same two men from the same lens), eight (the lodge table, 9:00–
   9:47), eight (#298–#304, the dining tent, 19:13–19:45), fourteen
   (#402–#416, the cremation ghat, 1.5 s each, which reads as flicker).
   Short cuts between near-identical frames of one shot are the worst
   kind of cut: all the cost of a cut, none of the information.
3. **Every 360 shot looks the same way.** All 38 slots from 360
   recordings are rendered at yaw 0, which on a selfie stick is the
   holder's face: #117–#121, #128–#138, #162–#166, #181–#183, #193–#201
   are a man looking at the camera with the landscape behind him. The
   sphere holds the trail, the companions and the mountains at other
   yaws, never shown. The `chosen_yaw` column exists and is empty: the
   framing step that was to fill it needs the captioning model.
4. **The summit swell is too loud.** The two Act 4 windows measure
   −7.9 and −9.5 LUFS integrated against −13 to −16 for every other
   window, −15 to −17 for the lines and −14 for the film, with the film
   peaking at −0.4 dBFS. It will clip on a phone speaker. Elsewhere the
   location-only stretches sit at −24 to −28 LUFS, ten LU under the
   lines: cinema-wide dynamics, too wide for a laptop.
5. **Three tracks in 77 s.** Act 5's first window (23:00–24:17) plays
   Home, then We Didn't Start the Fire, then 1.6 s of Start Me Up; the
   next window opens on Start Me Up again. Inside one window the
   assignment should not switch tracks, and a track once played should
   not return (the operator's ruling of 07:24).
6. **Spoken lines cut off.** Several speech slots end before the sentence
   does: the retime cuts a talking shot at the band's length, not at the
   end of the utterance (operator, 07:24; visible as the out-frame of
   #130, #136, #248 mid-word).
7. **No title, no credits, an abrupt end.** The film opens on a card and
   closes at 25:56 on a round video and a chat card. There is no name on
   the film, no date, no place, no credits, and the credits track in the
   config (`music.credits_track`) is never rendered.
8. **The opening image is the weakest frame in the film.** The cold open
   (#0, 0:00–0:16) is a portrait phone clip of the pass, a strip in the
   middle of black, held 16 s. The sphere recordings of the same morning
   exist (#330–#334).
9. **Static dialogue scenes run long.** Two lodge-interior sequences
   (9:00–9:47 and 19:09–19:45) are 40 s each of a seated man and a
   thermos, with the film's slowest cutting. They are where attention
   drops.
10. **Act 5 is a sprint.** Descent, Kathmandu, Pashupatinath, Delhi, the
    flight: 3.6 min for two weeks, against 16 min for the trek. Delhi is
    six slots. Either it is the coda and should be 90 s, or it is the
    return and needs its own line.

## Enhancements, ranked, each with the technical solution

1. **Fill the frame behind portrait and round clips.** In
   `render.segment_filters`, when the source's display shape is taller
   than the frame: `split[a][b]; [a]scale=960:540:force_original_aspect_
   ratio=increase,crop=960:540,boxblur=24:8[bg]; [b]scale=-2:540[fg];
   [bg][fg]overlay=(W-w)/2:0` — the blurred, filled copy of the same
   frame behind the clip, the standard treatment for vertical phone
   footage. For the Telegram round videos the same, with the circle's
   corners from the source alpha if present. Config: `render.portrait_fill:
   blur | black`. Cost: render-only, no box time beyond one cut.
2. **Hold the run rule to the end.** After `_chronological` in
   `build_timeline`, merge consecutive slots from the same recording whose
   source spans are contiguous into one slot (one cut instead of eight),
   and where they are not contiguous, enforce `max_consecutive_recording`
   by dropping the extra slots and closing the act up (the refill may
   then bring another recording). Function: a `_merge_runs(slots)` beside
   `_chronological`. The cremation ghat becomes two or three slots of six
   seconds; the jeep sequence one take with its line. Cost: a cut.
3. **Look somewhere other than the holder.** Until the captioning model
   fills `chosen_yaw`, a cheap heuristic in S03.6: for each 360 shot,
   sample four yaw stills (0, 90, 180, 270; `reproject.yaw_still_command`
   already does this), score each by motion energy and by not containing
   the largest face, and store the best as `chosen_yaw`; alternate
   between the holder (yaw 0) and the scene for consecutive 360 slots.
   The render already honours `chosen_yaw`. Cost: ~40 min on the box
   (4 stills × 1,100 360 shots), ~0.20 USD.
4. **Mix levels.** `music.placement.swell_lufs` down to −12; `render.
   location_full_lufs` from −18 to −16 and `location_under_speech_lufs`
   from −24 to −22, so the quiet stretches sit six LU under the lines,
   not ten; keep the −14 target and −1.5 dBTP ceiling. Cost: a cut.
5. **One track per window, one play per track.** In
   `music.assign_scenes`, a window is one assignment: all its scenes take
   the scene with the lowest cost's track; a track already laid in an
   earlier window gets `repeat_penalty` raised to a refusal
   (`music.one_play_per_track: true`). A segment switching track with
   less than `loop_min_piece_s` left in the window extends the previous
   cue instead. Cost: a cut.
6. **A line runs to its end.** In the retime, a video slot whose shot has
   speech ends at the end of the transcript segment the slot is in
   (`shots.transcript_json` has the segment times), bounded by the shot's
   end; the beat slots already hold through the line. Function: a clamp
   in `rhythm.retime` beside `_slot_avail_s`, reading `transcript_json`.
   Cost: a cut.
7. **Title and credits.** A title card after the cold open ("Манаслу.
   Апрель 2024." — the operator's words to choose) through the existing
   card path (`is_card`, `drawtext`), and a closing credits card over the
   last 20 s with the credits track (`music.credits_track`, never used):
   names are the operator's call; the pipeline has the two authors and
   Dharma the guide. Cost: a cut.
8. **A landscape cold open.** Prefer a landscape recording for the cold
   open beat's picture (`_cold_open`: rank candidates by aspect first),
   or fill behind it (enhancement 1).
9. **Shorter static scenes.** Cap a scene classed `resting`/`village`
   with no music at `assemble.static_scene_max_s` (25 s) in `plan_act`,
   dropping its surplus slots: the two lodge sequences halve.
10. **Act 5 as coda.** Either raise `acts[5]` toward 300 s with its own
    beat (a line about coming back) or accept 90 s and let Delhi be two
    stills and the plane.

## The pace-versus-length decision

Keep the eight windows and the music pace; the operator liked exactly
this. Recover the lost minutes from enhancement 2 (merging same-recording
runs frees shots and makes longer takes without changing the band
lengths) and from a slightly wider drift bound for Act 5 only
(`max_time_drift_h` 48 → 96 for the return, where the footage is weeks
apart). Do not slow the in-window bands: the operator asked for the
burst, not for less of it.

## What needs the operator's taste, not code

The title and credit texts; whether the cremation at Pashupatinath
belongs in a holiday film at all (it is 26 s of the current cut);
whether the lodge dialogue scenes are the film's rest or its drag; which
360 yaw to prefer when the heuristic is unsure (holder or scene);
whether the film ends on the chat card joke or on the mountain.
