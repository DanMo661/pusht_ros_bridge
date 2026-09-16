#!/bin/bash
# Chunk-native transport 10-episode benchmark: PNG lossless + action_chunk_size=8,
# same seeds 1000-1009 as the step-by-step jpeg/png/cli baselines.
set -e
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH

PY=/root/lerobot-venv/bin/python
SRC=/root/ros2_ws/src/pusht_ros_bridge/pusht_ros_bridge
BENCH=${BENCH:-/root/chunk_bench}
CHUNK=${CHUNK:-8}
rm -rf $BENCH && mkdir -p $BENCH

for SEED in $(seq 1000 1009); do
    $PY $SRC/policy_node.py --ros-args -p action_chunk_size:=$CHUNK \
        > /tmp/chunk_policy_$SEED.log 2>&1 &
    POLICY_PID=$!
    timeout 400 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
        -p image_codec:=png -p stats_path:=$BENCH/ep_$SEED.json \
        > /tmp/chunk_env_$SEED.log 2>&1 || true
    kill $POLICY_PID 2>/dev/null || true
    sleep 1
    cat $BENCH/ep_$SEED.json 2>/dev/null || echo "FAILED: $SEED"
done

$PY - <<EOF
import json, glob
rows = [json.load(open(f)) for f in sorted(glob.glob('$BENCH/ep_*.json'))]
if not rows:
    print('NO RESULTS'); raise SystemExit(1)
succ = sum(r['success'] for r in rows)
lat = [r['mean_step_latency_s'] for r in rows if r['mean_step_latency_s']]
wall = [r['episode_wall_s'] for r in rows]
rt = [r.get('obs_round_trips') or r['steps'] for r in rows]
print(f"CHUNK$CHUNK/PNG: episodes={len(rows)} success={succ}/{len(rows)} "
      f"({100*succ/len(rows):.0f}%) "
      f"mean_sum={sum(r['sum_reward'] for r in rows)/len(rows):.1f} "
      f"mean_wall={sum(wall)/len(wall):.1f}s "
      f"mean_round_trips={sum(rt)/len(rt):.1f} "
      f"mean_latency={sum(lat)/len(lat):.3f}s")
for r in rows:
    print(f"  seed={r['seed']} steps={r['steps']} sum={r['sum_reward']} "
          f"max={r['max_reward']} success={r['success']} wall={r['episode_wall_s']}s "
          f"trips={r.get('obs_round_trips')}")
EOF
echo CHUNK_BENCH_DONE
