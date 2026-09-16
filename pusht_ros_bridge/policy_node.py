"""LeRobot policy node: subscribes to PushT observations, runs the pretrained
diffusion policy on GPU, publishes actions.

ROS graph:
  sub  /pusht/observation  pusht_ros_bridge/PushtObservation (image + agent pos, atomic)
  pub  /pusht/action       std_msgs/Float32MultiArray (2,)

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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import PushtEnv
from lerobot.policies import make_policy, make_pre_post_processors

from pusht_ros_bridge.msg import PushtObservation

DEFAULT_MODEL = '/root/diffusion_pusht_migrated'


class PushTPolicyNode(Node):
    def __init__(self):
        super().__init__('pusht_policy')
        self.declare_parameter('model_path', DEFAULT_MODEL)

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

        self.action_pub = self.create_publisher(Float32MultiArray, '/pusht/action', 5)
        self.obs_sub = self.create_subscription(
            PushtObservation, '/pusht/observation', self.on_obs, 5)

        self._inference_count = 0

    # ── observation → action (one message = one complete observation) ──
    def on_obs(self, msg: PushtObservation):
        try:
            self._infer_and_publish(msg)
        except Exception:
            # A transient inference failure (CUDA hiccup, bad frame) must not
            # kill the node — the env has its own timeout fallback.
            self.get_logger().exception('inference failed, dropping observation')

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
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(ob)))

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
        if self._inference_count % 100 == 0:
            self.get_logger().info(f'inferences={self._inference_count}')


def main(args=None):
    rclpy.init(args=args)
    node = PushTPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
