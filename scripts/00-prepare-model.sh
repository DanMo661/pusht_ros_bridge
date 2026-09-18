#!/bin/bash
# Prepare a lerobot 0.6.x-compatible checkpoint from an old-format Hub model.
#
# The migration tool shipped in lerobot <=0.6.1 crashes inside save_pretrained
# (issue #4649: tuple-typed config fields hit draccus.encode), so this script
# tolerates a crash: processor files are taken from whatever the tool managed
# to write, while config.json + model.safetensors ALWAYS come from the pristine
# HF snapshot. Everything is assembled in a scratch directory and moved into
# OUT atomically, so an interrupted run can never corrupt an existing OUT.
#
# Usage: MODEL_REPO=lerobot/diffusion_pusht OUT=/root/model bash 00-prepare-model.sh
set -e
MODEL_REPO=${MODEL_REPO:-lerobot/diffusion_pusht}
OUT=${OUT:-/root/diffusion_pusht_migrated}
PY=${PY:-/root/lerobot-venv/bin/python}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # CN mirror; unset elsewhere
export HF_HOME=${HF_HOME:-/root/hf-cache}

SCRATCH="${OUT}.scratch"
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

# 1. official migration — produces the pre/post-processor files; may crash (see header)
"$PY" -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path "$MODEL_REPO" --output-dir "$SCRATCH" \
    || echo "NOTE: migration tool crashed (known issue #4649) — assembling from snapshot"

# 2. pristine config + weights: local dir or HF snapshot
if [ -d "$MODEL_REPO" ]; then
    cp "$MODEL_REPO/config.json" "$MODEL_REPO/model.safetensors" "$SCRATCH/"
    echo "copied config.json + model.safetensors from $MODEL_REPO"
else
    "$PY" - "$MODEL_REPO" "$SCRATCH" <<'PY'
import sys, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], Path(sys.argv[2])
snap = Path(snapshot_download(repo_id=repo, allow_patterns=["config.json", "model.safetensors"]))
for name in ("config.json", "model.safetensors"):
    src = snap / name
    assert src.is_file(), f"{name} missing from snapshot"
    shutil.copy(src, out / name)
    print(f"copied {name} ({src.stat().st_size} bytes)")
PY
fi

# 3. sanity check before touching OUT
"$PY" - "$SCRATCH" <<'PY'
import sys, json, os
out = sys.argv[1]
cfg = json.load(open(f"{out}/config.json"))
assert cfg.get("type"), "config.json has no policy type"
assert os.path.getsize(f"{out}/model.safetensors") > 1_000_000, "weights look wrong"
for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    assert os.path.getsize(f"{out}/{name}") > 0, f"{name} missing — migration produced nothing at all"
print("assembled model dir OK:", out)
PY

# 4. swap into place via renames only: rm+mv would leave a ~1 s window in
#    which an interrupted run destroys an existing OUT.
if [ -e "$OUT" ]; then
    mv "$OUT" "${OUT}.old"
fi
mv "$SCRATCH" "$OUT"
rm -rf "${OUT}.old"
echo "PREPARE_MODEL_DONE"
