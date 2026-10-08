#!/usr/bin/env bash
# Frozen 40+40 protected-image generation; method names retain proxy provenance.
set -uo pipefail
cd /data/MCAW || exit 2
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/traceguard/bin/python -u scripts/s20_generate_protected.py \
  --manifest data/formal_eval/fixed_40x2.jsonl \
  --naive-checkpoint outputs/s20_train_naive/checkpoints/final.pt \
  --coupled-checkpoint outputs/s20_train_coupled/checkpoints/final.pt \
  --output-root outputs/s20_images_40x2_proxy \
  --pgd-steps 10 \
  2>&1 | tee outputs/s20_images_40x2_proxy.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_images_40x2_proxy.exit
exit "$code"
