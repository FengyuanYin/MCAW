#!/usr/bin/env bash
# Generate matched Hallo videos for the four final-checkpoint image pilot samples.
set -uo pipefail
cd /data/MCAW || exit 2
export PATH=/root/miniconda3/envs/traceguard/bin:/root/miniconda3/bin:$PATH

input=outputs/s20_images_pilot2k/manifests
pilot=data/formal_eval/s20_video_pilot2k.jsonl
/root/miniconda3/envs/traceguard/bin/python - "$input" "$pilot" <<'PY'
import json
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])
methods = [
    'clean', 'wam', 'silencer_latent_proxy',
    'wam_then_silencer_latent_proxy', 'silencer_latent_proxy_then_wam',
    'naive_joint', 'message_coupled',
]
selected = []
reference = None
for method in methods:
    rows = [json.loads(line) for line in (source / f'{method}.jsonl').read_text().splitlines() if line.strip()]
    keys = [(row['domain'], row['sample_id'], row['audio']) for row in rows]
    if reference is None:
        reference = keys
        assert len(keys) == 4 and len({domain for domain, _, _ in keys}) == 2
        assert sorted([domain for domain, _, _ in keys]) == ['celeba', 'celeba', 'th1kh', 'th1kh']
    else:
        assert keys == reference, method
    selected.extend(rows)
destination.write_text(''.join(json.dumps(row) + '\n' for row in selected))
print(f'Validated {len(selected)} matched method/sample jobs', flush=True)
PY
if [[ $? -ne 0 ]]; then
  printf '2\n' > outputs/s20_video_pilot2k.exit
  exit 2
fi

/root/miniconda3/envs/traceguard/bin/python -u scripts/parallel_generate_videos.py \
  --manifest "$pilot" \
  --output-root outputs/s20_video_pilot2k \
  --gpus 3,4,5,6 \
  --min-free-mib 16000 \
  --video-name protected.mp4 \
  --traceguard /root/miniconda3/envs/traceguard/bin/traceguard \
  --ffprobe /root/miniconda3/bin/ffprobe \
  2>&1 | tee outputs/s20_video_pilot2k.log
code=${PIPESTATUS[0]}
printf '%s\n' "$code" > outputs/s20_video_pilot2k.exit
exit "$code"
