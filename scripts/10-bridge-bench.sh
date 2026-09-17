#!/bin/bash
# ROS 桥 10 回合基准：seed 1000-1009，每回合 policy+env 冷启动
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
BENCH=/root/ros_bench
rm -rf $BENCH && mkdir -p $BENCH

for SEED in $(seq 1000 1009); do
    echo "=== episode seed=$SEED ==="
    $PY $SRC/policy_node.py > /tmp/policy_$SEED.log 2>&1 &
    POLICY_PID=$!
    timeout 400 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
        -p stats_path:=$BENCH/ep_$SEED.json > /tmp/env_$SEED.log 2>&1 || true
    kill $POLICY_PID 2>/dev/null || true
    sleep 1
    cat $BENCH/ep_$SEED.json 2>/dev/null || echo "EPISODE FAILED: $SEED"
    echo ""
done

echo "=== 汇总 ==="
$PY - <<'EOF'
import json, glob
rows = []
for f in sorted(glob.glob('/root/ros_bench/ep_*.json')):
    rows.append(json.load(open(f)))
if not rows:
    print('NO RESULTS'); raise SystemExit(1)
succ = sum(r['success'] for r in rows)
print(f"episodes={len(rows)} success={succ}/{len(rows)} "
      f"({100*succ/len(rows):.0f}%)")
print(f"sum_reward: mean={sum(r['sum_reward'] for r in rows)/len(rows):.1f} "
      f"min={min(r['sum_reward'] for r in rows)} max={max(r['sum_reward'] for r in rows)}")
for r in rows:
    print(f"  seed={r['seed']} steps={r['steps']} sum={r['sum_reward']} "
          f"max={r['max_reward']} success={r['success']}")
EOF
echo ROS_BENCH_DONE
