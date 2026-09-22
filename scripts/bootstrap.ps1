$ErrorActionPreference = "Stop"
git submodule sync --recursive
git submodule update --init --recursive
python scripts/verify_upstreams.py

