#!/usr/bin/env bash
set -uo pipefail
cd /data/MCAW || exit 2
while [[ ! -f outputs/s20_video_pilot2k.exit ]]; do sleep 30; done
if [[ $(cat outputs/s20_video_pilot2k.exit) != 0 ]]; then
  printf '2\n' > outputs/s20_video_score2k.exit
  echo 'Video pilot failed; inspect outputs/s20_video_pilot2k.log' > outputs/s20_video_score2k.log
  exit 2
fi
CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/traceguard/bin/python -u scripts/s20_score_videos.py \
  --manifest data/formal_eval/s20_video_pilot2k.jsonl \
  --video-root outputs/s20_video_pilot2k \
  --output-root outputs/s20_video_score2k \
  2>&1 | tee outputs/s20_video_score2k.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_score2k.exit
exit "$code"
