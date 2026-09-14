#!/usr/bin/env bash
# Read-only deep dive on the camera folder and the music library.
# Answers: is the camera clock wrong (and by how much), does any recording
# actually chapter-split, what the .insv/.mp4 naming looks like, and what is
# in music/.
#
#   ./tools/survey_camera.sh /path/to/nepal_data
set -uo pipefail
ROOT="${1:-./nepal_data}"
D="$ROOT/media_from_camera"
[ -d "$D" ] || { echo "no media_from_camera under $ROOT"; exit 1; }
hr() { printf '%s\n' "----------------------------------------------------------------"; }

echo "CAMERA FILENAME DATE RANGE"; hr
ls "$D" | grep -oE '[0-9]{8}_[0-9]{6}' | sort | sed -n '1p;$p' | sed 's/^/  /'
echo
echo "  distinct dates in filenames:"
ls "$D" | grep -oE '_([0-9]{8})_' | tr -d _ | sort -u | tr '\n' ' ' | fold -w 60 | sed 's/^/    /'
echo; hr

echo "TRUE CHAPTER SPLITS  (one timestamp shared by several files)"; hr
shared=$(ls "$D" | grep -oE '[0-9]{8}_[0-9]{6}' | sort | uniq -c | awk '$1>1')
if [ -n "$shared" ]; then
  echo "$shared" | head -20 | sed 's/^/  /'
  echo "  -> these must collapse into one recording each"
  ts=$(echo "$shared" | head -1 | awk '{print $2}')
  echo "  files sharing $ts:"
  ls "$D" | grep "$ts" | sed 's/^/    /'
else
  echo "  none -- every file has a unique timestamp, so the trailing number is"
  echo "  a card-wide sequence counter, not a per-recording chapter index."
fi
hr

echo "NAMING BY EXTENSION"; hr
for e in insv lrv mp4 mov; do
  n=$(ls "$D" | grep -ci "\.$e$" || true)
  [ "$n" -gt 0 ] || continue
  echo "  .$e ($n files):"
  ls "$D" | grep -i "\.$e$" | head -5 | sed 's/^/    /'
done
hr

echo "THE TWO-DIGIT STREAM FIELD"; hr
ls "$D" | grep -oE '_[0-9]{2}_[0-9]{3}\.[A-Za-z]+$' |
  sed -E 's/_([0-9]{2})_[0-9]{3}\.(.*)$/\1 \2/' | sort | uniq -c | sort -rn | sed 's/^/  /'
hr

echo "EMBEDDED METADATA vs FILENAME  (is the clock wrong, or the name?)"; hr
if command -v exiftool >/dev/null 2>&1; then
  for e in insv mp4; do
    f=$(ls "$D"/*."$e" 2>/dev/null | head -1) || true
    [ -n "${f:-}" ] || continue
    echo "  $(basename "$f")"
    # plain tag listing, not -p: a -p format string prints nothing at all when
    # a tag is missing, and a missing CreateDate is precisely the finding here
    exiftool -CreateDate -MediaCreateDate -TrackCreateDate -Model -Duration \
             -ImageSize -GPSLatitude "$f" 2>/dev/null | sed 's/^/    /'
    exiftool -CreateDate "$f" 2>/dev/null | grep -q . || \
      echo "    (no CreateDate embedded -- only the filename carries a date)"
  done
  echo
  echo "  embedded CreateDate range across all camera video:"
  exiftool -q -r -d '%Y-%m-%d' -p '$CreateDate' -ext insv -ext mp4 -ext lrv "$D" 2>/dev/null |
    grep -E '^[0-9]{4}' | sort -u | sed -n '1p;$p' | sed 's/^/    /'
  echo
  echo "  phone video embedded CreateDate range (the comparison that matters):"
  exiftool -q -r -d '%Y-%m-%d' -p '$CreateDate' -ext mp4 -ext mov \
    "$ROOT/media_from_phones" 2>/dev/null |
    grep -E '^[0-9]{4}' | sort -u | sed -n '1p;$p' | sed 's/^/    /'
else
  echo "  exiftool not installed"
fi
hr

echo "MUSIC LIBRARY"; hr
M="$ROOT/music"
if [ -d "$M" ]; then
  find "$M" -type f | sed 's/^/  /'
  echo
  for f in "$M"/*.csv "$M"/*.txt; do
    [ -f "$f" ] || continue
    echo "  head of $(basename "$f"):"
    head -5 "$f" | sed 's/^/    /'
  done
  audio=$(find "$M" -type f \( -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.ogg' \) | wc -l)
  echo
  echo "  playable audio files: $audio"
  [ "$audio" -eq 0 ] && echo "  -> S02.7 has nothing to assign; acts cannot be scored yet"
fi
hr
