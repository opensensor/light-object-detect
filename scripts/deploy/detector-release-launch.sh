#!/usr/bin/env bash
set -euo pipefail
release_checkout="$HOME/.cache/opensensor-hosting/detector-release"
if [[ ! -d "$release_checkout/.git" ]]; then
    git clone --branch main --single-branch https://github.com/opensensor/light-object-detect.git "$release_checkout"
fi
git -C "$release_checkout" fetch origin main
git -C "$release_checkout" merge --ff-only origin/main
agent="$release_checkout/scripts/hosting_release.py"
[[ -f "$agent" ]] || exit 0
exec /usr/bin/python3 "$agent" --apply
