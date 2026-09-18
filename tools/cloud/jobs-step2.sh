#!/bin/bash
# The first real jobs on the box (Film v2 step 2, task 6), run detached so
# no ssh session or tool timeout can cut them short. Everything the local
# machine deferred while it was forbidden to compute: the tests that failed
# there, the manifest re-probe that brings in the heading tags and the new
# photos, the faces pass over the shots the EC2 pass never saw, and a fresh
# draft. Ends by pushing work/ to the bucket, which is what makes the
# results real for everybody else.
#
#   nepal remote exec -- 'nohup tools/cloud/jobs-step2.sh >/dev/null 2>&1 &'
#   ... then read /data/projects/nepal_work/reports/remote_jobs.log
set -uo pipefail
cd /data/projects/nepalvideo
N=.venv/bin/nepal
LOG=/data/projects/nepal_work/reports/remote_jobs.log
B=gs://nepalvideo-29922345852
t() { echo "--- $1 $(date -u +%T)"; }
{
echo "=== jobs start $(date -u +%FT%TZ) on $(hostname), $(nproc) cores"
t tests
.venv/bin/python -m pytest -q -p no:cacheprovider -m "slow or not slow" \
    tests/test_manifest_partial.py tests/test_e2e_s03.py tests/test_e2e_s02.py 2>&1 | tail -3
t s01
$N --no-progress s01 --redo manifest,chapters --skip-fov --skip-clock 2>&1 \
    | grep -E "S01\.[12]|S01 the manifest|WARN|ERROR|Traceback" | tail -14
t prune
$N prune 2>&1 | tail -4
t s02
$N --no-progress s02 --redo gps_track,geotag,acts 2>&1 \
    | grep -E "S02\.[1-4]|strava|trek window|WARN|ERROR|Traceback" | tail -10
t s03
$N --no-progress s03 --redo proxies,shots,photos,place,faces 2>&1 \
    | grep -E "S03\.[0-9]|S03 place|WARN|ERROR|Traceback" | grep -v "reading faces\|measuring\|detecting" | tail -20
t cut
$N --no-progress cut 2>&1 | grep -E "S05 |S06 |S07 |WARN|ERROR|Traceback" | tail -12
t push
gcloud storage rsync --recursive /data/projects/nepal_work "$B/work" 2>&1 | tail -1
echo "=== jobs done $(date -u +%FT%TZ)"
} > "$LOG" 2>&1
