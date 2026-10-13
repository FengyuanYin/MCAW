"""Invoke unchanged Hallo source with verified legacy Wav2Vec loading."""
import argparse
from pathlib import Path
import runpy
import sys

from traceguard.adapters.wav2vec_compat import install_verified_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hallo-root', type=Path, required=True)
    args, upstream = parser.parse_known_args()
    root = args.hallo_root.resolve()
    output = Path(upstream[upstream.index('--output') + 1]).resolve()
    sys.path.insert(0, str(root))
    install_verified_loader(output.parent / 'wav2vec_loading_audit.json')
    entry = root / 'scripts/inference.py'
    sys.argv = [str(entry), *upstream]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
