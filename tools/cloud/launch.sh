#!/bin/bash
# Launch the one rented GPU box. Run from the operator's machine with a live
# `aws login` session. Idempotent enough: re-running with an instance already
# tagged Name=nepal-pipeline prints its address and exits.
#
#   tools/cloud/launch.sh            # Spot (default)
#   SPOT=0 tools/cloud/launch.sh     # On-Demand, if Spot capacity or quota fails
#
# Everything created here is tagged Project=nepal so it can be found and
# deleted; the volume goes with the instance.
set -euo pipefail
REGION=eu-north-1
AMI="${AMI:-ami-0db574be841d285ac}"          # DL OSS Nvidia AMI PyTorch 2.7, Ubuntu 22.04, 2026-04-27
TYPE="${TYPE:-g4dn.xlarge}"
DISK_GB="${DISK_GB:-250}"
KEY=nepal-ec2
SG=$(aws ec2 describe-security-groups --region $REGION --filters Name=group-name,Values=nepal-pipeline --query 'SecurityGroups[0].GroupId' --output text)
HERE=$(cd "$(dirname "$0")" && pwd)

existing=$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Name,Values=nepal-pipeline Name=instance-state-name,Values=pending,running \
  --query 'Reservations[].Instances[].{id:InstanceId,ip:PublicIpAddress,state:State.Name}' --output text)
if [ -n "$existing" ]; then echo "already up: $existing"; exit 0; fi

market=()
if [ "${SPOT:-1}" = "1" ]; then
  market=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi

ID=$(aws ec2 run-instances --region $REGION \
  --image-id "$AMI" --instance-type "$TYPE" --key-name $KEY \
  --security-group-ids "$SG" --iam-instance-profile Name=nepal-pipeline-ec2 \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --user-data "file://$HERE/bootstrap.sh" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nepal-pipeline},{Key=Project,Value=nepal}]' \
                       'ResourceType=volume,Tags=[{Key=Project,Value=nepal}]' \
  "${market[@]}" \
  --query 'Instances[0].InstanceId' --output text)
echo "launched $ID (${SPOT:-1}==1 ? spot : on-demand)"
aws ec2 wait instance-running --region $REGION --instance-ids "$ID"
IP=$(aws ec2 describe-instances --region $REGION --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "running at $IP"
echo "ssh -i ~/.ssh/nepal-ec2 ubuntu@$IP   # bootstrap log: /var/log/nepal-bootstrap.log; done when /data/projects/BOOTSTRAP_DONE exists"
