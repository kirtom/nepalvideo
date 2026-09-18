#!/bin/bash
# GCE startup script for the nepal box (Film v2 section 2.2).
#
# Runs on EVERY boot (that is what a GCE startup script does), so every step
# is idempotent: the toolchain installs once behind a marker, the repo is
# reset to the branch each time, the bucket pull is an rsync. It never starts
# a stage: `nepal remote run` does that over SSH once /data/projects/READY
# exists. Log: /var/log/nepal-bootstrap.log.
set -euxo pipefail
exec > >(tee -a /var/log/nepal-bootstrap.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true; }
PROFILE="$(md nepal-profile)"; PROFILE="${PROFILE:-cpu}"
BRANCH="$(md nepal-branch)";   BRANCH="${BRANCH:-main}"
BUCKET="$(md nepal-bucket)"
REPO="${NEPAL_REPO:-https://github.com/kirtom/nepalvideo.git}"
USER_NAME=nepal
ROOT=/data/projects
rm -f $ROOT/READY

# -- toolchain, once ---------------------------------------------------
if [ ! -f $ROOT/.toolchain ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q ffmpeg libimage-exiftool-perl git python3-venv python3-pip \
      sqlite3 build-essential
  id -u $USER_NAME >/dev/null 2>&1 || useradd -m -s /bin/bash $USER_NAME
  mkdir -p $ROOT && chown -R $USER_NAME:$USER_NAME /data
  touch $ROOT/.toolchain
fi

# -- the Ops Agent, once: ships the stage logs and host metrics ---------
if [ ! -f $ROOT/.ops-agent ]; then
  curl -sSo /tmp/add-ops-agent.sh \
    https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
  bash /tmp/add-ops-agent.sh --also-install || echo "ops agent install failed; continuing"
  touch $ROOT/.ops-agent
fi

# -- GPU driver, once, only on the gpu profile ---------------------------
if [ "$PROFILE" = "gpu" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
  curl -sSLo /tmp/install_gpu_driver.py \
    https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
  python3 /tmp/install_gpu_driver.py || echo "driver install reported an error; continuing"
fi

# -- the repo, tracking the branch --------------------------------------
if [ ! -d $ROOT/nepalvideo/.git ]; then
  sudo -u $USER_NAME git clone --branch "$BRANCH" "$REPO" $ROOT/nepalvideo
fi
cd $ROOT/nepalvideo
sudo -u $USER_NAME git fetch -q origin "$BRANCH"
sudo -u $USER_NAME git reset -q --hard "origin/$BRANCH"
[ -d .venv ] || sudo -u $USER_NAME python3 -m venv .venv
sudo -u $USER_NAME .venv/bin/pip install -q --upgrade pip
sudo -u $USER_NAME .venv/bin/pip install -q -e '.[vision,music,asr,faces,semantic,dev]'

# -- the agent's config, every boot: it lives in the repo and may change --
if [ -f tools/cloud/ops-agent.yaml ] && [ -d /etc/google-cloud-ops-agent ]; then
  cp tools/cloud/ops-agent.yaml /etc/google-cloud-ops-agent/config.yaml
  systemctl restart google-cloud-ops-agent || true
fi

# -- the API key, from metadata into the user's environment -------------
KEY="$(md anthropic-api-key)"
if [ -n "$KEY" ]; then
  grep -q ANTHROPIC_API_KEY /home/$USER_NAME/.profile 2>/dev/null || \
    echo "export ANTHROPIC_API_KEY=$KEY" >> /home/$USER_NAME/.profile
fi

# -- the bucket -> the same paths the config names -----------------------
if [ -n "$BUCKET" ]; then
  # All three as the user, not root: the first boot made nepalvideo/data as
  # root, the rsync into it ran as nepal, and READY was never written.
  mkdir -p $ROOT/nepal_data $ROOT/nepal_work $ROOT/nepalvideo/data
  chown -R $USER_NAME:$USER_NAME $ROOT/nepal_data $ROOT/nepal_work $ROOT/nepalvideo/data
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/raw"  $ROOT/nepal_data
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/work" $ROOT/nepal_work
  sudo -u $USER_NAME gcloud storage rsync --recursive "$BUCKET/ref/data" $ROOT/nepalvideo/data
fi

sudo -u $USER_NAME .venv/bin/nepal doctor > $ROOT/doctor.txt 2>&1 || true
nvidia-smi -L >> $ROOT/doctor.txt 2>&1 || echo "no GPU visible" >> $ROOT/doctor.txt
date -u +%FT%TZ > $ROOT/READY
