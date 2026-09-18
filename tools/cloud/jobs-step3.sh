#!/bin/bash
# Film v2 step 3 on the box, detached. The whole suite (the fast tests on
# the box for the first time with the SDK installed, and the slow ones);
# the beat-sheet prompt written and priced before anything is re-made;
# then the re-transcription with word timestamps, which is the long part
# (about 0.7x realtime on 8 cores over two hours of speech), with the
# hallucination filter and the gate following inside S03; then the prompt
# again, now with cut points inside the shots. No live call: that needs
# the anthropic-api-key metadata, which the operator sets.
#
#   nepal remote exec -- 'nohup tools/cloud/jobs-step3.sh >/dev/null 2>&1 &'
#   ... then read /data/projects/nepal_work/reports/remote_jobs.log
set -uo pipefail
cd /data/projects/nepalvideo
N=.venv/bin/nepal
WORK=/data/projects/nepal_work
LOG=$WORK/reports/remote_jobs.log
FULL=$WORK/reports/remote_jobs
B=gs://nepalvideo-29922345852
mkdir -p "$FULL"

stage() {  # stage <name> <grep pattern> -- <command...>
  local name=$1 pat=$2; shift 2; [ "$1" = "--" ] && shift
  echo "--- $name $(date -u +%T)"
  "$@" > "$FULL/$name.log" 2>&1
  local rc=$?
  grep -E "$pat" "$FULL/$name.log" | grep -v "reading faces\|measuring\|detecting\|placing assets\|transcribing" | tail -${TAIL:-14}
  if grep -q "Traceback" "$FULL/$name.log"; then
    echo "!!! $name raised; the traceback:"; grep -A 12 "Traceback" "$FULL/$name.log" | tail -14
  fi
  echo "    ($name exit $rc, $(date -u +%T))"
}

MODE=${1:-full}       # full | resume: the corrected chain | beats: the live call
{
echo "=== jobs start $(date -u +%FT%TZ) on $(hostname), $(nproc) cores, step 3 ($MODE)"
if [ "$MODE" = "beats" ]; then
# The live call, detached like everything else: an ssh session that drops
# mid-answer would take the answer with it. Needs ANTHROPIC_API_KEY, which
# the bootstrap puts in ~/.profile from the anthropic-api-key metadata.
. ~/.profile 2>/dev/null
[ -n "${ANTHROPIC_API_KEY:-}" ] || echo "!!! ANTHROPIC_API_KEY is not set on this box (no anthropic-api-key metadata at boot?)"
TAIL=30 stage beats "S04\.5|WARN|ERROR" -- $N --no-progress beats
elif [ "$MODE" = "full" ]; then
TAIL=4 stage tests "passed|failed|error" -- \
  .venv/bin/python -m pytest -q -p no:cacheprovider -m "slow or not slow"
TAIL=8 stage beats-dry-1 "S04\.5|WARN|ERROR" -- $N --no-progress beats --dry-run
# The filter and the gate run inside S03 after the transcription; every
# other sub-step resumes through the data and costs nothing.
TAIL=24 stage s03 "S03\.[0-9]|S03 place|WARN|ERROR" -- $N --no-progress s03 --redo asr
else
# The camera clock by operator override (+14 d, not the solver's +18), the
# assets geotagged again at their real moments, the shots moved with their
# recordings by the place step, and the 218 transcriptions the status
# whitelist skipped -- shots_to_transcribe now owes them. The suite again,
# since the code changed.
TAIL=4 stage tests "passed|failed|error" -- \
  .venv/bin/python -m pytest -q -p no:cacheprovider -m "slow or not slow"
TAIL=10 stage s01 "S01\.5|clock|WARN|ERROR" -- $N --no-progress s01 --redo clock --skip-fov
TAIL=10 stage s02 "S02\.[1-4]|trek window|WARN|ERROR" -- \
  $N --no-progress s02 --redo geotag,acts
TAIL=24 stage s03 "S03\.[0-9]|S03 place|WARN|ERROR" -- $N --no-progress s03
fi
if [ "$MODE" != "beats" ]; then
TAIL=8 stage beats-dry-2 "S04\.5|WARN|ERROR" -- $N --no-progress beats --dry-run
fi
echo "--- push $(date -u +%T)"
$N status-page >/dev/null 2>&1
gcloud storage rsync --recursive "$WORK" "$B/work" 2>&1 | tail -1
echo "=== jobs done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
