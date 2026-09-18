#!/bin/bash
# The first real jobs on the box (Film v2 step 2, task 6), run detached so
# no ssh session or tool timeout can cut them short. Everything the local
# machine deferred while it was forbidden to compute: the tests that failed
# there, the manifest re-probe that brings in the heading tags and the new
# photos, the faces pass over the shots the EC2 pass never saw, and a fresh
# draft. Ends by pushing work/ to the bucket, which is what makes the
# results real for everybody else.
#
# Every stage's full output goes to its own file under reports/remote_jobs/;
# the summary log keeps the lines worth reading. A grep-filtered log once hid
# the traceback that explained everything, so the full text is always kept.
#
#   nepal remote exec -- 'nohup tools/cloud/jobs-step2.sh >/dev/null 2>&1 &'
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
  grep -E "$pat" "$FULL/$name.log" | grep -v "reading faces\|measuring\|detecting\|placing assets" | tail -${TAIL:-14}
  if grep -q "Traceback" "$FULL/$name.log"; then
    echo "!!! $name raised; the traceback:"; grep -A 12 "Traceback" "$FULL/$name.log" | tail -14
  fi
  echo "    ($name exit $rc, $(date -u +%T))"
}

{
echo "=== jobs start $(date -u +%FT%TZ) on $(hostname), $(nproc) cores"
TAIL=3 stage tests "passed|failed|error" -- \
  .venv/bin/python -m pytest -q -p no:cacheprovider -m "slow or not slow" \
    tests/test_manifest_partial.py tests/test_e2e_s03.py tests/test_e2e_s02.py
stage s01 "S01\.[12]|S01 the manifest|WARN|ERROR" -- \
  $N --no-progress s01 --redo manifest,chapters --skip-fov --skip-clock
TAIL=6 stage prune "prune|owns|produced" -- $N prune
stage s02 "S02\.[1-4]|strava|trek window|WARN|ERROR" -- \
  $N --no-progress s02 --redo gps_track,geotag,acts
TAIL=20 stage s03 "S03\.[0-9]|S03 place|WARN|ERROR" -- \
  $N --no-progress s03 --redo proxies,shots,photos,place,faces
stage cut "S05 |S06 |S07 |WARN|ERROR" -- $N --no-progress cut
echo "--- push $(date -u +%T)"
gcloud storage rsync --recursive "$WORK" "$B/work" 2>&1 | tail -1
echo "=== jobs done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
