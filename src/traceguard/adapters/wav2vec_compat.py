"""Explicitly load and verify legacy Wav2Vec weight-normalization tensors."""
from pathlib import Path


def verify_checkpoint_parameters(model, source):
    import torch
    source = {key[len('wav2vec2.'):] if key.startswith('wav2vec2.') else key: value for key, value in source.items()}
    converted, aliases = {}, {}
    for key, value in model.state_dict().items():
        original = key.replace('.parametrizations.weight.original0', '.weight_g').replace('.parametrizations.weight.original1', '.weight_v')
        if key in source:
            original = key
        if original in source:
            if source[original].shape != value.shape:
                raise ValueError(f'Wav2Vec checkpoint shape mismatch: {key}')
            converted[key] = source[original]
            if key != original:
                aliases[key] = original
    missing = sorted(set(model.state_dict()) - set(converted))
    if set(missing) - {'masked_spec_embed'}:
        raise ValueError(f'Wav2Vec critical parameters missing: {missing}')
    model.load_state_dict(converted, strict=False)
    model.eval()
    for key, value in converted.items():
        if not torch.equal(model.state_dict()[key].cpu(), value.cpu()):
            raise ValueError(f'Wav2Vec parameter verification failed: {key}')
    return dict(compatibility_aliases=aliases, verified_parameters=len(converted), unused_missing=missing)


def install_verified_loader(audit_path):
    import hashlib
    import json
    from safetensors.torch import load_file
    from hallo.models.wav2vec import Wav2VecModel
    original = Wav2VecModel.from_pretrained

    def verified(path, *args, **kwargs):
        return_info = kwargs.get('output_loading_info', False)
        kwargs['output_loading_info'] = True
        kwargs['local_files_only'] = True
        model, info = original(path, *args, **kwargs)
        checkpoint = Path(path) / 'model.safetensors'
        report = verify_checkpoint_parameters(model, load_file(str(checkpoint), device='cpu'))
        digest = hashlib.sha256()
        with checkpoint.open('rb') as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(block)
        report.update(initial_loading_info=info, model_sha256=digest.hexdigest())
        audit_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('WAV2VEC_CHECKPOINT_VERIFIED', report['verified_parameters'], flush=True)
        return (model, info) if return_info else model

    Wav2VecModel.from_pretrained = staticmethod(verified)
