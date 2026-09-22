from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from traceguard.config import LoadedConfig, load_config
from traceguard.preflight import checks_as_dicts, preflight_ok, run_preflight


BASELINES = ("wam", "silencer", "wam_then_silencer", "silencer_then_wam", "naive_joint", "message_coupled")


def _config(args: argparse.Namespace) -> LoadedConfig:
    return load_config(args.config, getattr(args, "overrides", None))


def _model(args: argparse.Namespace):
    from traceguard.factory import build_protector, load_protector_checkpoint, resolve_device

    config = _config(args)
    device = resolve_device(config)
    protector = build_protector(config, device)
    load_protector_checkpoint(protector, getattr(args, "checkpoint", None))
    protector.eval()
    return config, device, protector


def command_preflight(args: argparse.Namespace) -> int:
    checks = run_preflight(_config(args))
    for check in checks:
        print(f"{'OK' if check.ok else 'FAIL':4} {check.name:28} {check.detail}")
    if args.json:
        print(json.dumps(checks_as_dicts(checks), indent=2))
    return 0 if preflight_ok(checks) else 1


def command_verify_upstreams(_: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[3] / "scripts" / "verify_upstreams.py"
    return subprocess.run([sys.executable, str(script)], check=False).returncode


def command_protect(args: argparse.Namespace) -> int:
    import torch

    from traceguard.evaluation import image_metrics
    from traceguard.utils.images import load_image, message_to_string, parse_message, save_image

    config, device, protector = _model(args)
    image = load_image(args.image, int(config.require("model.image_size"))).to(device)
    message = parse_message(args.message, int(config.require("model.message_bits")), device)
    with torch.no_grad():
        output = protector(image, message)
    save_image(output.protected_images, args.output)
    result = image_metrics(image, output.protected_images)
    result.update({
        "message": message_to_string(output.decoded.bits[0]),
        "detection_score": float(output.decoded.detection_scores[0]),
        "output": str(Path(args.output).resolve()),
    })
    print(json.dumps(result, indent=2))
    return 0


def command_decode_image(args: argparse.Namespace) -> int:
    import torch

    from traceguard.utils.images import load_image, message_to_string

    _, device, protector = _model(args)
    image = load_image(args.image).to(device)
    with torch.no_grad():
        decoded = protector.wam.decode(image)
    print(json.dumps({
        "message": message_to_string(decoded.bits[0]),
        "bit_probabilities": decoded.bit_probabilities[0].cpu().tolist(),
        "bit_confidences": decoded.bit_confidences[0].cpu().tolist(),
        "detection_score": float(decoded.detection_scores[0]),
    }, indent=2))
    return 0


def command_decode_video(args: argparse.Namespace) -> int:
    from traceguard.evaluation import TemporalDecoder

    config, device, protector = _model(args)
    decoder = TemporalDecoder(protector.wam, float(config.get("evaluation.detection_threshold", 0.5)))
    result = decoder.decode_video(
        args.video,
        int(config.get("evaluation.frame_stride", 4)),
        int(config.get("evaluation.max_frames", 32)),
        str(device),
    )
    payload = result.as_dict()
    if args.message:
        target = args.message.strip().replace(" ", "")
        if len(target) != len(result.message) or any(bit not in "01" for bit in target):
            raise ValueError(f"--message must contain exactly {len(result.message)} binary digits")
        payload["target_message"] = target
        payload["bit_accuracy"] = sum(a == b for a, b in zip(result.message, target)) / len(target)
    print(json.dumps(payload, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    from traceguard.engine import Trainer

    trainer = Trainer(_config(args))
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run()
    return 0


def command_generate_video(args: argparse.Namespace) -> int:
    from traceguard.adapters.hallo_inference import generate_hallo_video

    generate_hallo_video(_config(args), Path(args.image), Path(args.audio), Path(args.output))
    print(json.dumps({"output": str(Path(args.output).resolve())}, indent=2))
    return 0


def command_reencode_video(args: argparse.Namespace) -> int:
    from traceguard.evaluation.reencode import reencode_h264

    output = reencode_h264(args.video, args.output, args.crf)
    print(json.dumps({"output": str(output), "crf": args.crf}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    from traceguard.baselines import generate_baseline
    from traceguard.data.manifest import read_manifest
    from traceguard.evaluation import image_metrics
    from traceguard.evaluation.metrics import bit_accuracy
    from traceguard.factory import build_proxy
    from traceguard.utils.images import load_image, parse_message
    from traceguard.utils.runtime import append_jsonl, seed_everything

    config, device, protector = _model(args)
    seed_everything(int(config.get("project.seed", 42)))
    proxy = build_proxy(config, device)
    methods = args.baseline or list(BASELINES)
    results_path = config.resolve("project.output_dir") / "evaluation.jsonl"
    for item in read_manifest(args.manifest):
        image = load_image(item.image, int(config.require("model.image_size"))).to(device)
        message = parse_message(item.message, int(config.require("model.message_bits")), device)
        for method in methods:
            output = generate_baseline(method, image, message, protector, proxy, args.pgd_steps)
            record = {
                "sample_id": item.sample_id,
                "method": method,
                **image_metrics(image, output.protected_images),
                "bit_accuracy": bit_accuracy(output.decoded.bits, message),
                "detection_score": float(output.decoded.detection_scores.detach().mean()),
            }
            append_jsonl(results_path, record)
            print(json.dumps(record, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traceguard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(name: str, function):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--set", dest="overrides", action="append", default=[])
        command.set_defaults(function=function)
        return command

    preflight = common("preflight", command_preflight)
    preflight.add_argument("--json", action="store_true")
    verify = subparsers.add_parser("verify-upstreams")
    verify.set_defaults(function=command_verify_upstreams)
    protect = common("protect", command_protect)
    protect.add_argument("--image", required=True)
    protect.add_argument("--message", required=True)
    protect.add_argument("--output", required=True)
    protect.add_argument("--checkpoint")
    decode_image = common("decode-image", command_decode_image)
    decode_image.add_argument("--image", required=True)
    decode_image.add_argument("--checkpoint")
    decode_video = common("decode-video", command_decode_video)
    decode_video.add_argument("--video", required=True)
    decode_video.add_argument("--checkpoint")
    decode_video.add_argument("--message")
    train = common("train", command_train)
    train.add_argument("--resume")
    video = common("generate-video", command_generate_video)
    video.add_argument("--image", required=True)
    video.add_argument("--audio", required=True)
    video.add_argument("--output", required=True)
    reencode = subparsers.add_parser("reencode-video")
    reencode.add_argument("--video", required=True)
    reencode.add_argument("--output", required=True)
    reencode.add_argument("--crf", type=int, default=23)
    reencode.set_defaults(function=command_reencode_video)
    evaluate = common("evaluate", command_evaluate)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--baseline", action="append", choices=BASELINES)
    evaluate.add_argument("--pgd-steps", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        status = args.function(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
