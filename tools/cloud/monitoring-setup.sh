#!/bin/bash
# Creates the error metric, the email channel, the alert and the dashboard
# (Film v2 step 2b). Idempotent: each is looked up by name first. Run from
# the operator's machine with a live gcloud login; nothing here computes.
#
#   tools/cloud/monitoring-setup.sh
#   NEPAL_ALERT_EMAIL=someone@example.com tools/cloud/monitoring-setup.sh
set -uo pipefail
G=${GCLOUD:-$HOME/google-cloud-sdk/bin/gcloud}
P=${NEPAL_PROJECT:-nepalvideo}
EMAIL=${NEPAL_ALERT_EMAIL:-$($G config get-value account 2>/dev/null)}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== log-based metric nepal_errors =="
if $G logging metrics describe nepal_errors --project "$P" >/dev/null 2>&1; then
  echo "exists"
else
  $G logging metrics create nepal_errors --project "$P" \
    --description "ERROR or Traceback lines in the pipeline's stage logs" \
    --log-filter "logName=\"projects/$P/logs/nepal_pipeline\" AND (severity>=ERROR OR textPayload:\"Traceback\" OR jsonPayload.message:\"Traceback\" OR jsonPayload.message:\"ERROR\")" 2>&1 | tail -1
fi

echo "== notification channel 'nepal email' -> $EMAIL =="
CH=$($G beta monitoring channels list --project "$P" --filter "displayName='nepal email'" --format 'value(name)' 2>/dev/null | head -1)
if [ -z "$CH" ]; then
  CH=$($G beta monitoring channels create --project "$P" --display-name 'nepal email' \
       --type email --channel-labels "email_address=$EMAIL" --format 'value(name)' 2>/dev/null)
  echo "created $CH"
else
  echo "exists $CH"
fi

echo "== alert policy 'nepal pipeline error' =="
POL=$($G alpha monitoring policies list --project "$P" --filter "displayName='nepal pipeline error'" --format 'value(name)' 2>/dev/null | head -1)
if [ -z "$POL" ]; then
  $G alpha monitoring policies create --project "$P" --policy-from-file "$HERE/alert-errors.json" \
     --notification-channels "$CH" 2>&1 | tail -1
else
  echo "exists $POL"
fi

echo "== dashboard 'nepal' =="
DASH=$($G monitoring dashboards list --project "$P" --filter "displayName='nepal'" --format 'value(name)' 2>/dev/null | head -1)
if [ -z "$DASH" ]; then
  $G monitoring dashboards create --project "$P" --config-from-file "$HERE/dashboard.json" 2>&1 | tail -1
else
  echo "exists $DASH; updating from the file"
  $G monitoring dashboards update "${DASH##*/}" --project "$P" --config-from-file "$HERE/dashboard.json" 2>&1 | tail -1
fi
echo "dashboards: https://console.cloud.google.com/monitoring/dashboards?project=$P"
