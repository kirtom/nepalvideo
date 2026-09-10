#!/usr/bin/env bash
# Read-only survey of nepal_data/. Writes nothing, changes nothing.
#
# Answers, in one pass: how big the corpus really is, whether .insv originals
# exist or only .lrv proxies, what the camera's chapter naming looks like,
# how many phone photos carry GPS, and whether the chat export contains a
# route file.
#
#   ./tools/survey_data.sh /path/to/nepal_data
set -uo pipefail
ROOT="${1:-./nepal_data}"
[ -d "$ROOT" ] || { echo "not a directory: $ROOT"; exit 1; }

hr() { printf '%s\n' "----------------------------------------------------------------"; }

echo "SURVEY OF $ROOT"; hr
echo "TOTAL SIZE"
du -sh "$ROOT" 2>/dev/null
echo
echo "BY TOP-LEVEL FOLDER"
du -sh "$ROOT"/*/ 2>/dev/null | sort -hr
hr

echo "BY EXTENSION  (count / total size)"
find "$ROOT" -type f 2>/dev/null | sed 's/.*\.//' | tr 'A-Z' 'a-z' | sort | uniq -c | sort -rn |
while read -r n ext; do
  sz=$(find "$ROOT" -type f -iname "*.${ext}" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.2f", s/1073741824}')
  printf "  %-8s %5d files  %8s GB\n" ".$ext" "$n" "$sz"
done
hr

echo "360 SOURCES  (spec section 12 Q1: do the originals exist?)"
n_insv=$(find "$ROOT" -type f -iname '*.insv' 2>/dev/null | wc -l)
n_lrv=$(find "$ROOT" -type f -iname '*.lrv'  2>/dev/null | wc -l)
n_mp4c=$(find "$ROOT" -path '*camera*' -type f -iname '*.mp4' 2>/dev/null | wc -l)
echo "  .insv originals : $n_insv"
echo "  .lrv proxies    : $n_lrv"
echo "  .mp4 flat/wide  : $n_mp4c"
if [ "$n_insv" -eq 0 ] && [ "$n_lrv" -gt 0 ]; then
  echo "  => PROXY ONLY. The 360 half caps near 1080p and S08 stitching is moot."
elif [ "$n_insv" -gt 0 ]; then
  echo "  => Originals present. Full-quality stitching is on the table."
fi
hr

echo "CAMERA FILENAMES  (first 12 -- these drive chapter grouping)"
find "$ROOT" -path '*camera*' -type f 2>/dev/null | xargs -r -n1 basename | sort | head -12 | sed 's/^/  /'
hr

echo "PHONE PHOTOS"
for who in keller kulikov; do
  n=$(find "$ROOT" -path "*phones/$who*" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.heic' -o -iname '*.png' -o -iname '*.dng' \) 2>/dev/null | wc -l)
  v=$(find "$ROOT" -path "*phones/$who*" -type f \( -iname '*.mp4' -o -iname '*.mov' \) 2>/dev/null | wc -l)
  printf "  %-9s %5d photos  %4d videos\n" "$who" "$n" "$v"
done
if command -v exiftool >/dev/null 2>&1; then
  echo "  GPS coverage (the geolocation spine for the whole project):"
  for who in keller kulikov; do
    d="$(find "$ROOT" -type d -path "*phones/$who" 2>/dev/null | head -1)"
    [ -n "$d" ] || continue
    tot=$(exiftool -q -r -n -if '$GPSLatitude' -p 1 "$d" 2>/dev/null | wc -l)
    all=$(exiftool -q -r -p 1 "$d" 2>/dev/null | wc -l)
    printf "    %-9s %d of %d files carry a fix\n" "$who" "$tot" "$all"
  done
  echo "  Date range:"
  exiftool -q -r -d '%Y-%m-%d' -p '$DateTimeOriginal' "$ROOT/media_from_phones" 2>/dev/null |
    grep -E '^[0-9]{4}' | sort -u | sed -n '1p;$p' | sed 's/^/    /'
else
  echo "  (install exiftool for GPS coverage and date range)"
fi
hr

echo "CHAT EXPORT  (spec section 12 Q2: is there a real route file?)"
ce="$ROOT/chat_export"
if [ -d "$ce" ]; then
  [ -f "$ce/result.json" ] && printf "  result.json     %s\n" "$(du -h "$ce/result.json" | cut -f1)"
  for sub in files photos round_video_messages video_files voice_messages; do
    [ -d "$ce/$sub" ] && printf "  %-22s %5d files\n" "$sub/" "$(find "$ce/$sub" -type f | wc -l)"
  done
  echo "  route files:"
  found=$(find "$ce" -type f \( -iname '*.gpx' -o -iname '*.kml' -o -iname '*.kmz' -o -iname '*.fit' \) 2>/dev/null)
  [ -n "$found" ] && echo "$found" | sed 's/^/    /' || echo "    none -- the GPS track will be built from photo EXIF"
else
  echo "  no chat_export/ found"
fi
hr

echo "MUSIC"
m="$ROOT/music"
[ -d "$m" ] && { printf "  %d tracks, %s\n" "$(find "$m" -type f | wc -l)" "$(du -sh "$m" | cut -f1)";
  find "$m" -type f | xargs -r -n1 basename | head -10 | sed 's/^/    /'; } || echo "  no music/ found"
hr

if command -v exiftool >/dev/null 2>&1; then
  echo "TOTAL VIDEO DURATION  (drives every cost estimate downstream)"
  exiftool -q -r -n -p '$Duration' -ext insv -ext mp4 -ext mov -ext lrv "$ROOT" 2>/dev/null |
    awk '{s+=$1; n++} END {
      if (!n) { print "  (none read)"; exit }
      if (s < 3600) printf "  %d clips, %.1f minutes\n", n, s/60;
      else printf "  %d clips, %.1f hours (%.0f min)\n", n, s/3600, s/60
    }'
fi
