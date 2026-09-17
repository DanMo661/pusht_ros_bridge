"""PushT environment node: runs the gym env, publishes observations, consumes actions.

ROS graph:
  pub  /pusht/observation  pusht_ros_bridge/PushtObservation (image + agent pos, atomic)
  sub  /pusht/action       std_msgs/Float32MultiArray (2,)

The episode loop runs on the main thread (blocking on actions) while a
background thread spins rclpy to service callbacks.
"""

import queue
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from pusht_ros_bridge.msg import PushtObservation
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import gymnasium as gym
import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
import cv2

# Handshake: re-publish the first observation until the policy node answers,
# for at most this long (model loading of a 1 GB checkpoint can be slow).
HANDSHAKE_INTERVAL_S = 2.0
HANDSHAKE_TIMEOUT_S = 180.0
ACTION_TIMEOUT_S = 60.0


class PushTEnvNode(Node):
    MAX_STEPS = 300

    def __init__(self):
        super().__init__('pusht_env')
        self.declare_parameter('seed', 7)
        self.declare_parameter('video_path', '')
        self.declare_parameter('stats_path', '')
        self.declare_parameter('image_codec', 'jpeg')  # 'jpeg' (lossy) or 'png' (lossless)

        self.obs_pub = self.create_publisher(PushtObservation, '/pusht/observation', 5)
        # Depth 32 matches the publisher: a chunk burst delivers up to N
        # messages back-to-back and none of them may be dropped mid-chunk.
        self.action_sub = self.create_subscription(
            Float32MultiArray, '/pusht/action', self.on_action, 32)

        # FIFO mailbox: actions queue up across the handshake burst instead of
        # collapsing to the latest one (matches the diffusion policy's own
        # 8-step action-queue semantics).
        self._action_q = queue.Queue(maxsize=32)

        self.env = gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos',
                            render_mode='rgb_array')
        seed = self.get_parameter('seed').value
        codec = self.get_parameter('image_codec').value
        if codec not in ('jpeg', 'png'):
            # Fail fast: an unrecognized value must not silently degrade to lossy jpeg.
            raise ValueError(f"invalid image_codec {codec!r}, use 'jpeg' or 'png'")
        self.get_logger().info(f'PushT env ready, will reset with seed={seed}')
        self.frames = []
        self._round_trip_latencies = []

    # ── action intake (called on the spin thread) ──────────────
    def on_action(self, msg: Float32MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('action message shorter than 2 values, dropped')
            return
        action = np.array(msg.data, dtype=np.float32)[:2]
        try:
            self._action_q.put_nowait(action)
        except queue.Full:
            try:
                self._action_q.get_nowait()  # drop oldest
                self._action_q.put_nowait(action)
            except queue.Empty:
                pass

    # ── helpers ────────────────────────────────────────────────
    def publish_obs(self, obs, stamp=None):
        codec = self.get_parameter('image_codec').value
        ext = '.png' if codec == 'png' else '.jpg'
        encode_params = ([cv2.IMWRITE_JPEG_QUALITY, 90] if ext == '.jpg'
                         else [cv2.IMWRITE_PNG_COMPRESSION, 1])
        rgb = obs['pixels']
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # cv2 encodes from BGR
        ok, buf = cv2.imencode(ext, bgr, encode_params)
        if not ok:
            self.get_logger().error(f'{codec} encode failed')
            return
        msg = PushtObservation()
        msg.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        img = CompressedImage()
        img.format = codec
        img.data = buf.tobytes()
        msg.image = img
        msg.agent_pos = [float(v) for v in obs['agent_pos']]
        self.obs_pub.publish(msg)

    def wait_action(self, timeout=ACTION_TIMEOUT_S):
        """Block for the next action from the policy node (None on timeout)."""
        try:
            return self._action_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── episode loop (main thread) ─────────────────────────────
    def run_episode(self):
        seed = self.get_parameter('seed').value
        obs, info = self.env.reset(seed=seed)
        self.frames.append(self.env.render())
        episode_start = time.time()

        # Handshake: heartbeat the first observation until the policy answers.
        # The first action received IS the inference result for the first obs.
        deadline = time.time() + HANDSHAKE_TIMEOUT_S
        action = None
        while time.time() < deadline:
            self.publish_obs(obs)
            action = self.wait_action(timeout=HANDSHAKE_INTERVAL_S)
            if action is not None:
                break
            self.get_logger().info('waiting for policy node...', throttle_duration_sec=10)
        if action is None:
            self.get_logger().error('policy node never answered, aborting')
            return None

        rewards, step, round_trips = [], 0, 0
        while step < self.MAX_STEPS:
            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(float(reward))
            step += 1
            self.frames.append(self.env.render())
            if terminated or truncated:
                break
            # Chunk-native cadence: only ask the policy for more actions when
            # the FIFO runs dry. With a chunking policy this cuts the message
            # rate by the chunk size; with a one-action policy (chunk=1) the
            # FIFO empties every step and the cadence is identical to before.
            try:
                action = self._action_q.get_nowait()
            except queue.Empty:
                self.publish_obs(obs)
                round_trips += 1
                # Time the observation round-trip (obs sent -> action received).
                # chunk=1: sampled every step; chunk=N: only at burst boundaries,
                # where the wait is the re-planning denoise pass.
                t_request = time.time()
                action = self.wait_action()
                self._round_trip_latencies.append(time.time() - t_request)
                if action is None:
                    self.get_logger().error('action timeout mid-episode, aborting')
                    return None
            if step % 100 == 0:
                self.get_logger().info(
                    f'step {step}/{self.MAX_STEPS} reward={reward:.3f} '
                    f'max={max(rewards):.3f}')

        wall_s = time.time() - episode_start
        result = {
            'seed': int(seed), 'steps': step,
            'sum_reward': round(sum(rewards), 2), 'max_reward': round(max(rewards), 4),
            'success': bool(info.get('is_success')),
            'episode_wall_s': round(wall_s, 2),
            'mean_round_trip_s': round(float(np.mean(self._round_trip_latencies)), 4)
            if self._round_trip_latencies else None,
            'obs_round_trips': round_trips + 1,  # +1: the handshake observation
            'codec': self.get_parameter('image_codec').value,
        }
        self.get_logger().info(f'EPISODE RESULT {result}')

        video_path = self.get_parameter('video_path').value
        if video_path and self.frames:
            h, w = self.frames[0].shape[:2]
            vw = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
            if not vw.isOpened():
                self.get_logger().error('VideoWriter failed to open, no video saved')
            else:
                for f in self.frames:
                    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                vw.release()
                self.get_logger().info(f'video saved: {video_path}')

        stats_path = self.get_parameter('stats_path').value
        if stats_path:
            import json
            with open(stats_path, 'w') as f:
                json.dump(result, f)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PushTEnvNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        result = node.run_episode()
    except KeyboardInterrupt:
        result = None
    finally:
        try:
            node.env.close()
        except Exception:
            pass
        # Spin must stop before the node is destroyed (rclpy requirement).
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()

    sys.exit(0 if result is not None else 1)


if __name__ == '__main__':
    main()
