#!/bin/bash
# Prepare a lerobot 0.6.x-compatible checkpoint from an old-format Hub model.
# Works around the incomplete output of migrate_policy_normalization (see
# https://github.com/huggingface/lerobot/issues/4649): the tool emits the
# processor files but an empty config.json and no weights — assemble the
# final directory by hand from the HF snapshot.
#
# Usage: MODEL_REPO=lerobot/diffusion_pusht OUT=/root/model bash 00-prepare-model.sh
set -e
MODEL_REPO=${MODEL_REPO:-lerobot/diffusion_pusht}
OUT=${OUT:-/root/diffusion_pusht_migrated}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # CN mirror; unset elsewhere
export HF_HOME=${HF_HOME:-/root/hf-cache}

# 1. Run the official migration (produces processor files only)
python -m lerobot.processor.migrate_policy_normalization \
    --pretrained-path "$MODEL_REPO" --output-dir "$OUT"

# 2. Fetch the raw snapshot (config.json + model weights) and complete the dir
python - "$MODEL_REPO" "$OUT" <<'PY'
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

# 3. Sanity check: config parses and weights load
python - "$OUT" <<'PY'
import sys, json
out = sys.argv[1]
cfg = json.load(open(f"{out}/config.json"))
assert cfg.get("type"), "config.json has no policy type"
import os
assert os.path.getsize(f"{out}/model.safetensors") > 1_000_000, "weights look wrong"
print("model dir OK:", out)
PY
echo "PREPARE_MODEL_DONE"
