#!/bin/bash
# Run one episode: env node + policy node, both with the lerobot venv python
# (ros2 run forces the system python, which lacks the venv dependencies; rclpy
# and the generated message module are injected below).
# All paths are overridable: WS (colcon workspace), PY, MODEL, CHUNK, CODEC, BRIDGE_OUT.
set -e
WS=${WS:-/root/ros2_ws}
# Nodes run from the workspace copy (native Linux filesystem — running from a
# /mnt path goes through the slow 9P bridge). 08-bridge-build.sh rsyncs the
# clone into the workspace before every build, so this is the code you cloned.
SRC="$WS/src/pusht_ros_bridge/pusht_ros_bridge"
source /opt/ros/jazzy/setup.bash
if [ ! -f "$WS/install/setup.bash" ]; then
    echo "workspace $WS is not built — run scripts/08-bridge-build.sh first (WS=$WS)"
    exit 1
fi
source "$WS/install/setup.bash"   # generated PushtObservation msg module
if [ ! -d "$SRC" ]; then
    echo "package missing under $WS — run scripts/08-bridge-build.sh first"
    exit 1
fi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH

PY=${PY:-/root/lerobot-venv/bin/python}
OUT=${BRIDGE_OUT:-/root/ros_pusht_episode}
SEED=${1:-7}
MODEL=${MODEL:-/root/diffusion_pusht_migrated}
CHUNK=${CHUNK:-1}
CODEC=${CODEC:-jpeg}
rm -rf "$OUT" && mkdir -p "$OUT"

$PY $SRC/policy_node.py --ros-args -p model_path:=$MODEL -p action_chunk_size:=$CHUNK > $OUT/policy.log 2>&1 &
POLICY_PID=$!

# Wait for the policy to finish loading (log line), but bail out immediately
# if the process dies instead of blind-waiting for the env heartbeat timeout.
READY=""
for i in $(seq 1 90); do
    if grep -q "ready on" $OUT/policy.log 2>/dev/null; then READY=1; break; fi
    if ! kill -0 $POLICY_PID 2>/dev/null; then
        echo "POLICY_NODE_DIED_DURING_LOAD:"; tail -8 $OUT/policy.log; exit 1
    fi
    sleep 1
done
[ -n "$READY" ] || echo "policy not ready after 90 s, starting env anyway (heartbeat will wait)"

timeout 600 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
    -p video_path:=$OUT/ros_episode.mp4 -p stats_path:=$OUT/stats.json \
    -p image_codec:=$CODEC \
    > $OUT/env.log 2>&1 &
ENV_PID=$!
# Watchdog: a policy crash must not leave the env waiting for its full timeout.
while kill -0 $ENV_PID 2>/dev/null; do
    if ! kill -0 $POLICY_PID 2>/dev/null; then
        echo "POLICY_NODE_DIED_MID_EPISODE:"; tail -8 $OUT/policy.log
        kill $ENV_PID 2>/dev/null || true
        break
    fi
    sleep 2
done
wait $ENV_PID || true
kill $POLICY_PID 2>/dev/null || true
sleep 1

if [ ! -s "$OUT/stats.json" ]; then
    echo "=== env.log tail ==="
    tail -4 $OUT/env.log
    echo "EPISODE_FAILED (no stats.json)"
    exit 1
fi
echo "=== env.log tail ==="
tail -4 $OUT/env.log
echo "=== stats ==="
cat $OUT/stats.json
echo ""
echo "=== 产物 ==="
ls -la $OUT/
echo BRIDGE_RUN_DONE
