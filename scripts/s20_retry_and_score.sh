#!/usr/bin/env bash
# Resume only failed S20 pilot videos, then score the complete matched grid.
set -uo pipefail
cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH

/root/miniconda3/envs/traceguard/bin/python -u scripts/parallel_generate_videos.py \
  --manifest data/formal_eval/s20_video_pilot2k.jsonl \
  --output-root outputs/s20_video_pilot2k \
  --gpus 0 \
  --video-name protected.mp4 \
  --traceguard /root/miniconda3/envs/traceguard/bin/traceguard \
  --ffprobe /root/miniconda3/bin/ffprobe \
  2>&1 | tee outputs/s20_video_retry2k.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_retry2k.exit
if [[ "$code" != 0 ]]; then exit "$code"; fi

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/traceguard/bin/python -u scripts/s20_score_videos.py \
  --manifest data/formal_eval/s20_video_pilot2k.jsonl \
  --video-root outputs/s20_video_pilot2k \
  --output-root outputs/s20_video_score2k \
  2>&1 | tee outputs/s20_video_score2k.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_score2k.exit
exit "$code"
