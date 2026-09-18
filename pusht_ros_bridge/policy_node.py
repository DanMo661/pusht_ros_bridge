"""LeRobot policy node: subscribes to PushT observations, runs the pretrained
diffusion policy on GPU, publishes actions.

ROS graph:
  sub  /pusht/observation  pusht_ros_bridge/PushtObservation (image + agent pos, atomic)
  pub  /pusht/action       std_msgs/Float32MultiArray (2,) — one message per
                           action; with action_chunk_size=N, one observation
                           round-trip triggers a burst of N such messages

Must run with the lerobot venv interpreter (rclpy is injected via PYTHONPATH):
  source /root/lerobot-venv/bin/activate
  source /root/ros2_ws/install/setup.bash   # generated msg python module
  export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
  export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH
"""

import lerobot.scripts.lerobot_eval  # noqa: F401  populates the policy registry
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import cv2
import threading
import traceback

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.policies import make_policy, make_pre_post_processors

from pusht_ros_bridge.msg import PushtObservation

DEFAULT_MODEL = '/root/diffusion_pusht_migrated'


class PushTPolicyNode(Node):
    def __init__(self):
        super().__init__('pusht_policy')
        self.declare_parameter('model_path', DEFAULT_MODEL)
        # >1 turns on chunk-native transport: one observation round-trip
        # returns this many actions back-to-back (the env only asks when its
        # action FIFO runs dry). 1 = classic one-action-per-observation.
        self.declare_parameter('action_chunk_size', 1)
        # Real-time mode: inference moves off the ROS executor onto a worker
        # thread that always plans against the newest observation (receding
        # horizon). Pairs with the env node's refill_watermark>=0. False keeps
        # the classic synchronous request/response flow.
        self.declare_parameter('async_inference', False)

        model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'loading policy: {model_path}')
        cfg = PreTrainedConfig.from_pretrained(model_path)
        cfg.pretrained_path = model_path   # 0.6.1 does not backfill; without it
        # the processor silently builds with empty stats (lerobot #4647)
        cfg.device = 'cuda'
        self.policy = make_policy(cfg=cfg, env_cfg=PushtEnv())
        self.policy.eval()
        self.policy.reset()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={'device_processor': {'device': 'cuda'}})
        self.get_logger().info(f'{type(self.policy).__name__} ready on {cfg.device}')

        # Depth must cover a whole burst: with action_chunk_size=N the node
        # publishes N action messages back-to-back, and a shallow DDS history
        # silently drops the tail of the burst mid-chunk.
        self.action_pub = self.create_publisher(Float32MultiArray, '/pusht/action', 32)
        self.obs_sub = self.create_subscription(
            PushtObservation, '/pusht/observation', self.on_obs, 5)

        self._inference_count = 0
        self._last_obs_fingerprint = None

        # Receding-horizon plumbing (async_inference=true): on_obs only slots
        # the newest observation; the daemon worker turns it into a burst.
        self.async_mode = bool(self.get_parameter('async_inference').value)
        self._latest_obs = None
        self._pending_fp = None   # fingerprint of the newest stored observation
        self._planned_fp = None   # fingerprint last handed to inference
        self._obs_gate = threading.Condition()
        self._stop_evt = threading.Event()

    # ── observation → action (one message = one complete observation) ──
    def on_obs(self, msg: PushtObservation):
        try:
            # The env heartbeats an unanswered observation verbatim every few
            # seconds. In step mode a duplicate heartbeat just pops one more
            # action from the queue, but with action_chunk_size>1 each queued
            # duplicate triggers a FULL burst — the episode then starts with
            # several overlapping re-sampled chunks for the same observation,
            # which poisons the trajectory (measured: near-zero coverage on
            # seeds the direct loop solves). Deduplicate identical consecutive
            # observations; a genuine re-send is only answered once.
            fingerprint = (hash(bytes(msg.image.data)), tuple(msg.agent_pos))
            if self.async_mode:
                with self._obs_gate:
                    if fingerprint == self._pending_fp:
                        self.get_logger().debug('duplicate observation (heartbeat), skipped')
                        return
                    # Obs arriving while an inference is in flight collapse to
                    # the newest one: the plan that matters is the one against
                    # the freshest state (receding horizon).
                    self._latest_obs = msg
                    self._pending_fp = fingerprint
                    self._obs_gate.notify()
                return
            if fingerprint == self._last_obs_fingerprint:
                self.get_logger().debug('duplicate observation (heartbeat), skipped')
                return
            self._last_obs_fingerprint = fingerprint
            self._infer_and_publish(msg)
        except Exception:
            # A transient inference failure (CUDA hiccup, bad frame) must not
            # kill the node — the env has its own timeout fallback. rclpy's
            # logger supports neither .exception() nor exc_info=, so the
            # traceback goes into the message body.
            self.get_logger().error(
                'inference failed, dropping observation\n' + traceback.format_exc())

    def _planner_loop(self):
        """Async-mode worker: plan against the newest observation, forever."""
        while not self._stop_evt.is_set():
            with self._obs_gate:
                while not self._stop_evt.is_set() and self._pending_fp == self._planned_fp:
                    self._obs_gate.wait(timeout=1.0)
                if self._stop_evt.is_set():
                    return
                msg = self._latest_obs
                self._planned_fp = self._pending_fp
            try:
                self._infer_and_publish(msg)
            except Exception:
                if self._stop_evt.is_set() or not rclpy.ok():
                    return  # shutdown race (kill mid-plan), not a real failure
                # Re-arm the gate so the env's next heartbeat (same bytes)
                # passes the dedup check and the observation is retried once
                # per heartbeat instead of starving until the env's timeout.
                with self._obs_gate:
                    self._pending_fp = None
                    self._planned_fp = None
                self.get_logger().error(
                    'async inference failed, will retry on next heartbeat\n'
                    + traceback.format_exc())

    def request_stop(self):
        self._stop_evt.set()
        with self._obs_gate:
            self._obs_gate.notify_all()

    def _infer_and_publish(self, msg: PushtObservation):
        if msg.image.format not in ('jpeg', 'png'):
            self.get_logger().warn(f'unsupported image format {msg.image.format!r}, dropped')
            return
        pixels = cv2.imdecode(np.frombuffer(msg.image.data, dtype=np.uint8),
                              cv2.IMREAD_COLOR)
        if pixels is None:
            self.get_logger().error('image decode failed')
            return

        # Mirrors lerobot.envs.utils.preprocess_observation:
        # BGR (HWC uint8) -> CHW float [0,1] with a batch dimension.
        img = torch.from_numpy(pixels[:, :, ::-1].copy())
        img = img.permute(2, 0, 1).float().unsqueeze(0) / 255.0
        state = torch.tensor([float(v) for v in msg.agent_pos],
                             dtype=torch.float32).unsqueeze(0)

        ob = {'observation.image': img, 'observation.state': state}
        batch = self.pre(ob)
        # select_action re-infers only when its internal action queue is empty,
        # so a burst of chunk_size calls costs one denoising pass and pops the
        # whole queue — exactly the policy's own chunking semantics, just
        # transported in one round-trip instead of eight.
        #
        # Chunking without the intermediate observations makes the policy's
        # obs-history queue half stale at each re-inference ([obs_{t-N}, obs_t]
        # instead of [obs_{t-1}, obs_t]) — measured as a total quality collapse
        # on diffusion_pusht (max coverage 1.0 -> 0.02). Resetting before the
        # burst re-fills the history by duplicating the current observation
        # ([obs_t, obs_t]), the same well-conditioned pattern the policy sees
        # at episode start; measured quality returns to step-mode level
        # (max 1.0, sum 198 vs 205) with none of the collapse.
        chunk = max(1, int(self.get_parameter('action_chunk_size').value))
        with torch.inference_mode():
            if chunk > 1:
                self.policy.reset()
            for _ in range(chunk):
                action = self.post(self.policy.select_action(batch))
                # Never emit a bare scalar: flatten whatever shape the policy produced.
                values = np.atleast_1d(
                    action.to('cpu').numpy().astype(np.float32)).flatten().tolist()
                if len(values) < 2:
                    self.get_logger().error(f'action has {len(values)} values, need 2, dropped')
                    return

                out = Float32MultiArray()
                out.data = values
                self.action_pub.publish(out)
                self._inference_count += 1
        if self._inference_count % 100 < chunk:
            self.get_logger().info(f'inferences={self._inference_count}')


def main(args=None):
    rclpy.init(args=args)
    node = PushTPolicyNode()
    worker = None
    if node.async_mode:
        worker = threading.Thread(target=node._planner_loop, daemon=True)
        worker.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        # Normal exit path when the launcher stops us mid-spin; don't dump a trace.
        pass
    except rclpy.error.RCLError:
        # Shutdown can also race spin's wait-set creation (kill mid-plan).
        # Same normal exit path, but a live context means a real error.
        if rclpy.ok():
            raise
    finally:
        node.request_stop()
        rclpy.try_shutdown()
        if worker is not None:
            worker.join(timeout=2.0)
        node.destroy_node()


if __name__ == '__main__':
    main()
