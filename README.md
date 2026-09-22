# MCAW: Message-Conditioned Adversarial Watermarking for Traceable Talking-Head Immunization

Research prototype for learning one message-conditioned residual that both carries a WAM watermark and disrupts unauthorized audio-driven portrait animation.

## Design constraints

- WAM, Silencer, and Hallo are pinned Git submodules and are never edited.
- All integration code lives in `src/traceguard`.
- The default recipe targets one 24 GB RTX 3090: 256 px, batch size 1, AMP, frozen WAM detector and frozen talking-head proxy.
- Model weights and datasets are never downloaded implicitly.
- Only use portraits and audio that you own or are authorized to process.

## Setup on the remote server

```bash
git clone --recurse-submodules https://github.com/FengyuanYin/MCAW.git
cd MCAW
conda create -n traceguard python=3.10 -y
conda activate traceguard
pip install -r third_party/hallo/requirements.txt
pip install -e '.[wam,hallo,metrics]'
traceguard preflight --config configs/train_3090.yaml
```

If this working tree was copied without submodules, run `scripts/bootstrap.sh`. The script fetches source code only; it does not download checkpoints.

## Required weights

Set paths in `configs/train_3090.yaml`:

- WAM MIT checkpoint (`wam_mit.pth`) and its bundled `params.json`.
- Hallo pretrained model directory following the upstream layout.
- Optional LPIPS and SyncNet/LSE dependencies for full evaluation.

Prepare the official weights explicitly on the server:

```bash
mkdir -p checkpoints pretrained_models
wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P checkpoints/
huggingface-cli download fudan-generative-ai/hallo --local-dir pretrained_models
```

These are manual setup commands, not runtime behavior. Review the upstream licenses and storage requirements before running them.

Run `traceguard preflight` before training. Missing paths are reported with actionable messages and a non-zero exit status.

## Commands

```bash
traceguard preflight --config configs/train_3090.yaml
traceguard train --config configs/train_3090.yaml
traceguard protect --config configs/train_3090.yaml --image input.png --message 01010101010101010101010101010101 --output protected.png
traceguard decode-image --config configs/train_3090.yaml --image protected.png
traceguard decode-video --config configs/evaluate.yaml --video generated.mp4
traceguard evaluate --config configs/evaluate.yaml --manifest examples/authorized_manifest.jsonl
traceguard verify-upstreams
```

## Training stages

1. `watermark_warmup`: train the new message-conditioned residual adapter while preserving WAM decoding.
2. `joint_proxy`: enable the frozen Hallo/Silencer proxy and coupling loss.
3. `robust_finetune`: enable differentiable distortions.
4. Periodically run full Hallo inference under `torch.no_grad()`; it is never in the per-step backward path.

`backend: mock` is an engineering smoke backend. Research results must use `backend: real` and must record all missing/failed research thresholds.

The first real training stage uses Silencer's latent/reference-feature attack route (`silencer_latent`) because it is the path that fits reliably on one 3090. Full Hallo video generation is an independent no-gradient validation command; audio-control nullification claims must be based on those video metrics, not on the latent proxy loss alone.

`naive_joint` and `message_coupled` share the same inference architecture but must be trained with coupling weight 0 and non-zero respectively. Evaluate them in separate runs with their corresponding checkpoints; using one checkpoint for both is only an interface smoke check, not a valid ablation.

## Upstream integrity

Run:

```bash
python scripts/verify_upstreams.py
```

The command checks pinned commits and dirty status. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources and licenses.
