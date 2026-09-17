#!/bin/bash
# Build pusht_ros_bridge in a colcon workspace.
# The repo (this script's parent directory) is rsynced into $WS/src first, so
# the built code is always exactly what you cloned — never a stale copy.
set -e
set -o pipefail
WS=${WS:-/root/ros2_ws}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
mkdir -p "$WS/src"
rsync -a --delete --exclude '.git' "$REPO_DIR/" "$WS/src/pusht_ros_bridge/"
cd "$WS"
colcon build --packages-select pusht_ros_bridge 2>&1 | tail -3
test -f "$WS/install/pusht_ros_bridge/lib/pusht_ros_bridge/env_node.py"
echo BUILD_OK
