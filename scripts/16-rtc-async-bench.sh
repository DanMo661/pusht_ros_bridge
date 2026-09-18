#!/bin/bash
# Real-time (async receding-horizon) benchmark: policy worker thread
# (async_inference) + proactive FIFO refill (refill_watermark) on top of
# chunked execution. Same PNG codec and seeds 1000-1009 as 15-chunk-bench.sh,
# so rows are directly comparable to the chunk sweep.
# CHUNK / WATERMARK / ASYNC override the knobs; ASYNC=false + WATERMARK=-1
# reproduces the synchronous 15-chunk-bench.sh run exactly.
set -e
WS=${WS:-/root/ros2_ws}
PY=${PY:-/root/lerobot-venv/bin/python}
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH

SRC="$WS/src/pusht_ros_bridge/pusht_ros_bridge"
BENCH=${BENCH:-/root/rtc_bench}
CHUNK=${CHUNK:-8}
WATERMARK=${WATERMARK:-4}
ASYNC=$(echo "${ASYNC:-true}" | tr '[:upper:]' '[:lower:]')
[ "$ASYNC" = true ] && TAG="CHUNK$CHUNK+ASYNC wm$WATERMARK" || TAG="CHUNK$CHUNK+SYNC wm$WATERMARK"
rm -rf "$BENCH" && mkdir -p "$BENCH"

for SEED in $(seq 1000 1009); do
    $PY $SRC/policy_node.py --ros-args -p action_chunk_size:=$CHUNK -p async_inference:=$ASYNC \
        > /tmp/rtc_policy_$SEED.log 2>&1 &
    POLICY_PID=$!
    timeout 400 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
        -p image_codec:=png -p refill_watermark:=$WATERMARK -p stats_path:=$BENCH/ep_$SEED.json \
        > /tmp/rtc_env_$SEED.log 2>&1 || true
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
lat = [r.get('mean_round_trip_s') or r.get('mean_step_latency_s') for r in rows if r.get('mean_round_trip_s') or r.get('mean_step_latency_s')]
wall = [r['episode_wall_s'] for r in rows]
rt = [r.get('obs_round_trips') or r['steps'] for r in rows]
print(f"$TAG/PNG: episodes={len(rows)} success={succ}/{len(rows)} "
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
echo RTC_BENCH_DONE
