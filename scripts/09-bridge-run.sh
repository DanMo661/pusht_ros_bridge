#!/bin/bash
# 联调（自定义消息版）：env 节点 + policy 节点都用 lerobot venv python 直跑
# （ros2 run 强制系统 python 跑不了 venv 依赖；rclpy+msg 走 PYTHONPATH 注入）
set -e
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash   # 生成的 PushtObservation msg 模块
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export SDL_VIDEODRIVER=dummy
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH

PY=/root/lerobot-venv/bin/python
SRC=/root/ros2_ws/src/pusht_ros_bridge/pusht_ros_bridge
OUT=${BRIDGE_OUT:-/root/ros_pusht_episode}
SEED=${1:-7}
MODEL=${MODEL:-/root/diffusion_pusht_migrated}
CHUNK=${CHUNK:-1}
CODEC=${CODEC:-jpeg}
rm -rf $OUT && mkdir -p $OUT

$PY $SRC/policy_node.py --ros-args -p model_path:=$MODEL -p action_chunk_size:=$CHUNK > $OUT/policy.log 2>&1 &
POLICY_PID=$!

sleep 12  # 等 1GB 权重加载完

timeout 600 $PY $SRC/env_node.py --ros-args -p seed:=$SEED \
    -p video_path:=$OUT/ros_episode.mp4 -p stats_path:=$OUT/stats.json \
    -p image_codec:=$CODEC \
    > $OUT/env.log 2>&1 || true

kill $POLICY_PID 2>/dev/null || true
sleep 1
echo "=== env.log tail ==="
tail -4 $OUT/env.log
echo "=== stats ==="
cat $OUT/stats.json 2>/dev/null || echo "(no stats)"
echo "=== 产物 ==="
ls -la $OUT/
echo BRIDGE_RUN_DONE
