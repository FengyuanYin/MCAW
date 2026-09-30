#!/usr/bin/env bash
# Frozen 40+40 seven-input Hallo batch, including clean and six proxy/joint methods.
set -uo pipefail
cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH

root=outputs/s20_images_40x2_proxy/manifests
manifest=data/formal_eval/s20_video_40x2_proxy.jsonl
/root/miniconda3/envs/traceguard/bin/python - "$root" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
methods = [
    'clean', 'wam', 'silencer_latent_proxy',
    'wam_then_silencer_latent_proxy', 'silencer_latent_proxy_then_wam',
    'naive_joint', 'message_coupled',
]
fixed = [json.loads(line) for line in Path('data/formal_eval/fixed_40x2.jsonl').read_text().splitlines() if line.strip()]
fixed_ids = {row['sample_id'] for row in fixed}
assert len(fixed) == len(fixed_ids) == 80
reference = None
selected = []
for method in methods:
    rows = [json.loads(line) for line in (source / f'{method}.jsonl').read_text().splitlines() if line.strip()]
    keys = [(row['domain'], row['sample_id'], row['audio'], row['message']) for row in rows]
    assert len(rows) == 80 and {row['sample_id'] for row in rows} == fixed_ids
    assert sum(row['domain'] == 'celeba' for row in rows) == 40
    assert sum(row['domain'] == 'th1kh' for row in rows) == 40
    assert all(row['method'] == method for row in rows)
    if reference is None:
        reference = keys
    else:
        assert keys == reference, method
    selected.extend(rows)
destination.write_text(''.join(json.dumps(row) + '\n' for row in selected), encoding='utf-8')
print(f'Validated {len(selected)} jobs from {len(methods)} matched input sets', flush=True)
PY
if [[ $? -ne 0 ]]; then
  printf '2\n' > outputs/s20_video_40x2_proxy.exit
  exit 2
fi

/root/miniconda3/envs/traceguard/bin/python -u scripts/parallel_generate_videos.py \
  --manifest "$manifest" \
  --output-root outputs/s20_video_40x2_proxy \
  --gpus 0,3,4,5,6 \
  --min-free-mib 16000 \
  --video-name protected.mp4 \
  --traceguard /root/miniconda3/envs/traceguard/bin/traceguard \
  --ffprobe /root/miniconda3/bin/ffprobe \
  2>&1 | tee outputs/s20_video_40x2_proxy.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_40x2_proxy.exit
if [[ "$code" != 0 ]]; then exit "$code"; fi

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/traceguard/bin/python -u scripts/s20_score_videos.py \
  --manifest "$manifest" \
  --video-root outputs/s20_video_40x2_proxy \
  --output-root outputs/s20_video_score_40x2_proxy \
  2>&1 | tee outputs/s20_video_score_40x2_proxy.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_score_40x2_proxy.exit
exit "$code"
