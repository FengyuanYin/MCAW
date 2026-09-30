#!/usr/bin/env bash
# One protected-image Hallo compatibility check before the batch pilot.
set -uo pipefail
cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH
head -n 1 outputs/s20_images_pilot24/manifests/wam.jsonl > data/formal_eval/s20_wam_video_one.jsonl
/root/miniconda3/envs/traceguard/bin/python -u scripts/parallel_generate_videos.py \
  --manifest data/formal_eval/s20_wam_video_one.jsonl \
  --output-root outputs/s20_video_one \
  --gpus 5 \
  --video-name protected.mp4 \
  --traceguard /root/miniconda3/envs/traceguard/bin/traceguard \
  --ffprobe /root/miniconda3/bin/ffprobe \
  2>&1 | tee outputs/s20_video_one.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_one.exit
exit "$code"
