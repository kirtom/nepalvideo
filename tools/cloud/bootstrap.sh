#!/bin/bash
# cloud-init user-data for the one rented GPU box (docs/STATE.md, "cloud").
#
# Sets the machine up to run the same `nepal` CLI against the same paths as
# the operator's machine, so config/pipeline.yaml needs no change: the media
# subset lives at /data/projects/nepal_data and the work dir at
# /data/projects/nepal_work, pulled from S3. It does NOT start a stage: the
# database is synced last, after the local run has stopped, and the operator
# starts `nepal s03` over SSH once it is there.
#
# Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04): CUDA and the driver are
# present; ffmpeg, exiftool and the Python deps are not.
set -euxo pipefail
exec > /var/log/nepal-bootstrap.log 2>&1

BUCKET="${NEPAL_BUCKET:-nepalvideo-kk-eun1}"
BRANCH="${NEPAL_BRANCH:-claude/zen-carson-7iwfqo}"
REPO="https://github.com/kirtom/nepalvideo.git"

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q ffmpeg libimage-exiftool-perl git python3-venv python3-pip sqlite3

mkdir -p /data/projects
chown -R ubuntu:ubuntu /data
cd /data/projects
sudo -u ubuntu git clone --branch "$BRANCH" "$REPO" nepalvideo

cd nepalvideo
sudo -u ubuntu python3 -m venv .venv
# ctranslate2 (faster-whisper) wants CUDA 12 + cuDNN 9 as pip wheels even
# when the AMI has them system-wide; cheaper to install than to link.
sudo -u ubuntu .venv/bin/pip install -q --upgrade pip
sudo -u ubuntu .venv/bin/pip install -q -e '.[vision,music,asr,cloud,dev]' \
    nvidia-cublas-cu12 nvidia-cudnn-cu12

# the media subset and the work dir, at the paths the config already names
sudo -u ubuntu aws s3 sync "s3://$BUCKET/raw/"  /data/projects/nepal_data/ --only-show-errors
sudo -u ubuntu aws s3 sync "s3://$BUCKET/work/" /data/projects/nepal_work/ --only-show-errors

sudo -u ubuntu .venv/bin/nepal doctor > /data/projects/doctor.txt 2>&1 || true
nvidia-smi -L >> /data/projects/doctor.txt 2>&1 || echo "no GPU visible" >> /data/projects/doctor.txt
touch /data/projects/BOOTSTRAP_DONE
