# One atomic PushT observation: a frame image plus the pusher position,
# guaranteed to arrive together and in sync.
std_msgs/Header header
sensor_msgs/CompressedImage image
float32[2] agent_pos
