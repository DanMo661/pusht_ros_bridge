#!/bin/bash
# PNG 无损传输 10 回合对照（vs jpeg q90 基准），同 seed 1000-1009
set -e
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH

PY=/root/lerobot-venv/bin/python
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_DIR/pusht_ros_bridge"
BENCH=/root/png_bench
rm -rf $BENCH && mkdir -p $BENCH

for SEED in $(seq 1000 1009); do
    $PY $SRC/policy_node.py > /tmp/png_policy_$SEED.log 2>&1 &
    POLICY_PID=$!
    timeout 400 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
        -p image_codec:=png -p stats_path:=$BENCH/ep_$SEED.json \
        > /tmp/png_env_$SEED.log 2>&1 || true
    kill $POLICY_PID 2>/dev/null || true
    sleep 1
    cat $BENCH/ep_$SEED.json 2>/dev/null || echo "FAILED: $SEED"
done

$PY - <<'EOF'
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob('/root/png_bench/ep_*.json'))]
if not rows:
    print('NO RESULTS'); raise SystemExit(1)
succ = sum(r['success'] for r in rows)
lat = [r.get('mean_round_trip_s') or r.get('mean_step_latency_s') for r in rows if r.get('mean_round_trip_s') or r.get('mean_step_latency_s')]
print(f"PNG: episodes={len(rows)} success={succ}/{len(rows)} "
      f"({100*succ/len(rows):.0f}%) "
      f"mean_sum={sum(r['sum_reward'] for r in rows)/len(rows):.1f} "
      f"mean_latency={sum(lat)/len(lat):.3f}s")
for r in rows:
    print(f"  seed={r['seed']} steps={r['steps']} sum={r['sum_reward']} "
          f"max={r['max_reward']} success={r['success']} lat={r.get('mean_round_trip_s') or r.get('mean_step_latency_s')}")
EOF
echo PNG_BENCH_DONE
