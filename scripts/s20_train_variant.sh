#!/usr/bin/env bash
# Train the two joint-model ablations on the same expanded manifest and seed.
set -uo pipefail

variant=${1:?usage: s20_train_variant.sh naive|coupled GPU STEPS CHECKPOINT_EVERY [RESUME]}
gpu=${2:?GPU index required}
steps=${3:?training steps required}
checkpoint_every=${4:?checkpoint interval required}
resume=${5:-}

case "$variant" in
  naive) coupling=0 ;;
  coupled) coupling=0.1 ;;
  *) echo "variant must be naive or coupled" >&2; exit 2 ;;
esac
[[ "$gpu" =~ ^[0-9]+$ && "$steps" =~ ^[0-9]+$ && "$checkpoint_every" =~ ^[0-9]+$ ]] || exit 2

cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
out="outputs/s20_train_${variant}"
log="outputs/s20_train_${variant}.log"
exit_file="outputs/s20_train_${variant}.exit"
test -s data/train_expanded_v2.jsonl || exit 2
if [[ -z "$resume" && -e "$out/checkpoints/final.pt" ]]; then
  echo "Refusing to restart over existing checkpoint: $out/checkpoints/final.pt" >&2
  exit 2
fi
if [[ -n "$resume" && ! -s "$resume" ]]; then
  echo "Resume checkpoint missing: $resume" >&2
  exit 2
fi

command=(
  /root/miniconda3/envs/traceguard/bin/traceguard train
  --config configs/train_3090.yaml
  --set "project.output_dir=$out"
  --set train.manifest=data/train_expanded_v2.jsonl
  --set "train.steps=$steps"
  --set "train.checkpoint_every=$checkpoint_every"
  --set "loss.coupling=$coupling"
)
if [[ -n "$resume" ]]; then
  command+=(--resume "$resume")
fi
printf 'variant=%s gpu=%s steps=%s checkpoint_every=%s resume=%s\n' \
  "$variant" "$gpu" "$steps" "$checkpoint_every" "$resume" | tee -a "$log"
CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" 2>&1 | tee -a "$log"
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > "$exit_file"
exit "$code"
