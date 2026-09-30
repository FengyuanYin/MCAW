#!/usr/bin/env bash
# Resume the 40+40 proxy video batch after CUDA outage, preserving prior results.
set -uo pipefail
cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH

/root/miniconda3/envs/traceguard/bin/python -u scripts/parallel_generate_videos.py \
  --manifest data/formal_eval/s20_video_40x2_proxy.jsonl \
  --output-root outputs/s20_video_40x2_proxy \
  --gpus 3,4,5,6 \
  --min-free-mib 16000 \
  --video-name protected.mp4 \
  --traceguard /root/miniconda3/envs/traceguard/bin/traceguard \
  --ffprobe /root/miniconda3/bin/ffprobe \
  2>&1 | tee outputs/s20_video_40x2_proxy_resume.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_40x2_proxy_resume.exit
if [[ "$code" != 0 ]]; then exit "$code"; fi

CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/traceguard/bin/python -u scripts/s20_score_videos.py \
  --manifest data/formal_eval/s20_video_40x2_proxy.jsonl \
  --video-root outputs/s20_video_40x2_proxy \
  --output-root outputs/s20_video_score_40x2_proxy \
  2>&1 | tee outputs/s20_video_score_40x2_proxy.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_score_40x2_proxy.exit
exit "$code"
