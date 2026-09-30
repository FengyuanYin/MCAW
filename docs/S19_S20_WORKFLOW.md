# S19 and S20 workflow

This workflow requires explicitly prepared weights, authorized inputs, and the real backend for research results. Data, checkpoints, logs, and videos are excluded from Git. Merge the Hallo integration, S19 review, and training/baseline PRs before running S20.

## Review and freeze

After S18 clean generation finishes, run from the repository root:

```bash
python scripts/s19_prepare_review.py
python scripts/s19_review_server.py --port 8765
```

Forward port 8765 over SSH. The server binds to localhost and persists votes in `data/formal_eval/clean_review.csv`. Watch complete videos; do not select by protected-method performance. After at least 40 approved samples per domain:

```bash
python scripts/s19_freeze.py
```

This freezes 40 CelebA-HQ + 40 TH1KH samples with hashes and refuses to overwrite frozen files. Path overlap checks do not prove content or identity disjointness; check those during data preparation.

## Train paired checkpoints

Shell launchers assume `/data/MCAW` and `/root/miniconda3/envs/traceguard`. Inspect paths and GPU indices before running. Check `nvidia-smi` and choose available GPUs; memory checks do not reserve GPUs against other users.

```bash
bash scripts/s20_train_variant.sh naive 3 2000 500
bash scripts/s20_train_variant.sh coupled 4 2000 500
```

Use equal steps, manifest, seed, and configuration. Coupling is 0 versus 0.1. Append a checkpoint path as the fifth argument to resume. Check logs and `.exit` files before continuing. Resume restores optimizer and random state, but not shuffled data-loader position.

## Generate and verify images

```bash
python scripts/s20_generate_protected.py \
  --manifest data/formal_eval/fixed_40x2.jsonl \
  --naive-checkpoint outputs/s20_train_naive/checkpoints/final.pt \
  --coupled-checkpoint outputs/s20_train_coupled/checkpoints/final.pt \
  --output-root outputs/s20_images_40x2_proxy
python scripts/s20_verify_images.py \
  --root outputs/s20_images_40x2_proxy \
  --fixed-manifest data/formal_eval/fixed_40x2.jsonl
```

Image generation requires an empty output directory. Names containing `silencer_latent_proxy` identify a VAE latent-distance approximation, not official full Silencer. The verifier checks saved image budgets before video generation.

## Generate and resume videos

Inspect the GPU list in `scripts/s20_video_full_proxy.sh`, then run it to validate matched inputs, create the combined manifest, generate 560 videos, and score on success. The pilot launcher requires separately prepared four-sample pilot images.

Once the combined manifest exists, explicit GPU selection and resumption use:

```bash
python scripts/parallel_generate_videos.py \
  --manifest data/formal_eval/s20_video_40x2_proxy.jsonl \
  --output-root outputs/s20_video_40x2_proxy \
  --gpus 3,4,5 --min-free-mib 16000 --video-name protected.mp4
```

Restart with the same arguments after interruption. Successful statuses are skipped when method and input paths match. Each GPU runs one subprocess, with unique temporary video names to avoid MoviePy collisions. Workers stop for low free memory or known CUDA failures; failed or pending jobs give a nonzero exit. After reboot, recreate tmux and rerun the command.

## Score and inspect

```bash
python scripts/s20_score_videos.py \
  --manifest data/formal_eval/s20_video_40x2_proxy.jsonl \
  --video-root outputs/s20_video_40x2_proxy \
  --output-root outputs/s20_video_score_40x2_proxy
python scripts/s20_status.py
```

Scoring requires a complete method/sample grid and matching successful statuses. It reports watermark recovery and detection statistics. Lip-sync disruption, identity preservation, and video quality require separate measurements before making immunization claims. Status and CUDA audits are diagnostics, not proof that research thresholds passed.

## Regression checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests use CPU tensors, temporary inputs, mocked video subprocesses, and a localhost server. Real Hallo inference and GPU performance require remote validation.
