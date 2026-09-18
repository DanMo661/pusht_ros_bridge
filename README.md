# pusht_ros_bridge

Run a pretrained [LeRobot](https://github.com/huggingface/lerobot) diffusion policy as a **ROS 2 node pair** — observations flow in over topics, actions flow out over topics, GPU inference happens in between. A minimal working take on the first step of the [ROS 2 integration RFC (huggingface/lerobot#4368)](https://github.com/huggingface/lerobot/issues/4368): policy-as-a-node, validated end-to-end.

![](demo_episode.gif)

**Status: experimental — PushT simulation only, not wired to any real robot.**

Tested with: ROS 2 Jazzy · Python 3.12 · lerobot 0.6.1 · torch 2.11 + CUDA 12.x on an RTX 4060 Laptop GPU · Ubuntu 24.04 (WSL2) · RMW CycloneDDS.

## Architecture

```
┌──────────────────────┐                  ┌───────────────────────────┐
│       env_node       │   /pusht/        │        policy_node        │
│                      │   observation    │                           │
│   gym PushT-v0   ────┼─────────────────►│  jpeg/png decode          │
│   96x96 rgb +        │  (custom msg,    │  LeRobot preprocessing    │
│   agent pos          │   atomic)        │  diffusion policy (GPU)   │
│                      │                  │  postprocessing           │
│   episode loop  <────┼──────────────────┤                           │
│                      │   /pusht/action  │                           │
└──────────────────────┘   (2 floats)     └───────────────────────────┘
```

Design points:

- **`pusht_ros_bridge/PushtObservation`** — one message = one complete observation (`CompressedImage` + `float32[2]` agent position). Image and state arrive atomically; no cross-publisher ordering hazards, no "wait until both halves are here" bookkeeping.
- **Observation heartbeat handshake** — until the policy node answers, the env re-publishes the first observation every 2 s (up to 180 s), so an episode starts whenever the slow-loading 1 GB policy is ready. No startup race.
- **FIFO action mailbox** (bounded, drops oldest) — matches the diffusion policy's own action-queue semantics.
- Two independent processes; the only coupling is the two topics.

## Benchmark (10 episodes each, identical seeds 1000–1009)

Metric definitions — PushT **success**: T-block coverage ≥ 0.95 **held to the end** of a max-300-step episode (so a "failed" episode can still reach max coverage 0.99+ without holding it); **sum_reward**: cumulative per-step coverage, roughly 0–250 per episode. Seeds are matched across runs: the bridge loops seeds explicitly, and the lerobot CLI increments its seed per episode from `--seed` (verified in `lerobot_eval` source).

| Metric | Bridge, jpeg q90 | Bridge, PNG lossless | lerobot CLI (in-process) |
|---|---|---|---|
| Success rate | 4/10 (40%) | **7/10 (70%)** | 7/10 (70%) |
| Mean sum reward | 119.6 | 110.5 | 109.1 |
| Sum reward range | 28.4 – 250.2 | 28.9 – 240.9 | 27.2 – 253.8 |
| Mean per-step round-trip | 0.209 s† | 0.269 s | n/a (in-process) |
| Mean episode wall time | ~65 s† | 78.7 s | ~11 s* |

\* CLI time excludes process startup; the bridge numbers include per-episode process cold start (~12 s model load).
† jpeg round-trip measured on a single instrumented episode (latency instrumentation landed after the jpeg benchmark); jpeg wall time estimated from it as `steps × 0.21 s + 12 s`.

Reading the numbers:

- **The bridge itself costs nothing**: PNG (lossless) transport matches the CLI success rate exactly, 7/10, on identical seeds — topic hop + message serialization do not degrade the policy.
- **The jpeg q90 gap (40% vs 70%) is compression, not transport**: it is not statistically significant at n=10 (Fisher exact p ≈ 0.37), but the seed-level pattern is telling — e.g. seed 1001 scores max coverage 0.38 over jpeg and succeeds over PNG.
- Per-episode raw JSON for all 30 episodes (plus the VQ-BeT run): [`benchmarks/`](benchmarks/). Aggregated latency over the 10 PNG episodes: mean 269 ms/step (median 260 ms), mean wall 78.7 s; jpeg episode timings above are the single-episode instrumented estimate.

## Policy-agnostic check (VQ-BeT)

To verify the bridge is not tuned to one policy, it was also run with `lerobot/vqbet_pusht` (VQ-BeT — a codebook-based behavioral transformer, architecturally unrelated to diffusion), migrated with `lerobot.processor.migrate_policy_normalization` and passed via the `model_path` ROS parameter. On seed 1000 the bridge and a direct in-process loop score **identically** (sum 4.85, max coverage 0.0162), matching the diffusion result that the bridge is bit-exact relative to direct inference. The low absolute score is the checkpoint's own behavior under the current lerobot pipeline, not a transport effect (migrated weights were verified bit-identical to the original snapshot, minus the 6 extracted normalization buffers). Practical notes: the 151 MB VQ-BeT checkpoint loads in ~3 s and steps at ~13 ms round-trip (vs 269 ms for diffusion's 1 GB UNet on the same machine).

```bash
MODEL=/root/vqbet_pusht_migrated bash scripts/09-bridge-run.sh 1000
```

## Chunked execution (`action_chunk_size > 1`)

Chunking policies (diffusion, VQ-BeT) internally plan `n_action_steps` actions per denoising pass but the classic bridge still ships **one observation per step** — 7 of every 8 round-trips only pop an already-computed action from the policy's queue. With `action_chunk_size=N` the protocol turns chunk-native instead: the env requests observations **only when its action FIFO runs dry**, and the policy answers each request with a burst of N actions (one denoising pass, whole queue shipped). Sweep on the 10-seed PNG benchmark (seeds 1000–1009):

| chunk size | Success | Mean max coverage | Obs round-trips/episode |
|---|---|---|---|
| 1 (step-by-step) | **7/10** | **0.989** | ≈ its step count (mean 232) |
| 2 | 0/10 | 0.488 | 150 |
| 4 | 1/10 | 0.854 | 72.9 |
| 8 | 5/10 | 0.919 | 29.8 |
| *direct in-process chunk-8 (no ROS, reference)* | *2/10* | *0.85* | *0* |

Reading the numbers:

- **Round-trips scale exactly as 300/N** (measured: 150 / 72.9 / 29.8), and each burst boundary pays one denoising pass (~2 s for this policy on an RTX 4060), so denoising passes per episode scale as steps/N: the chunk-8 run averaged 29.8 passes and 81.0 s wall vs step-by-step's 232-step episodes averaging ~29 passes and 78.7 s — near-identical wall at N=8 — while smaller N re-plans far more often and pays proportionally more wall time (chunk-2 averaged 150 passes; its benchmark ran on a thermally throttled GPU, so its 298 s wall is not directly comparable). The transport win is that a burst boundary is the *only* place the control loop waits: on a bandwidth-limited or jittery link (real robot over Wi-Fi) this is the difference between a control loop that stalls and one that never waits.
- **Chunking costs success on this task, and the mechanism is feedback latency**: every action k..k+7 is planned from one observation, so the policy corrects N× less often. The failure signature is characteristic — episodes reach high coverage (0.96+ in several cases) but cannot *hold* it to the end, exactly the open-loop degradation that motivates re-conditioning schemes like Real-Time Chunking. The bridge matches the direct in-process chunk-8 baseline within sampling variance, so the transport adds nothing on top.
- **Differences among N∈{2,4,8} are not resolvable at n=10**: the diffusion policy samples stochastically and unseeded, and the N=2 row scoring *below* N=8 makes that plain — with ~10 draws of a bimodal policy the ordering inside the chunked regime is noise. What survives is the comparison against N=1 (every chunked point is clearly worse) and the exact round-trip scaling. Resolving a sweet spot would need seeded paired rollouts; treat N as a transport knob, keep N=1 when control quality matters, and validate on your own task.
- Three conditions must hold for the burst to work at all, all learned the hard way:
  1. **The policy's observation-history queue must be reset at each burst.** Without the intermediate observations, the queue at re-inference holds `[obs_{t-8}, obs_t]` instead of `[obs_{t-1}, obs_t]`, and diffusion_pusht collapses to near-zero coverage (1.0 → 0.02 on seeds it otherwise solves). Resetting re-fills the history with the current observation duplicated — `[obs_t, obs_t]` — the same well-conditioned pattern the policy sees at episode start; quality returns to step-mode level. Done automatically when `action_chunk_size > 1`.
  2. **Heartbeat duplicates must be deduplicated.** The env re-publishes an unanswered observation every 2 s; in step mode a queued duplicate just pops one more action (benign), but each queued duplicate triggering a full burst seeds the episode with overlapping re-sampled chunks (near-zero coverage again). The policy node fingerprints consecutive observations and answers each distinct one once.
  3. **DDS history must fit the burst.** Burst publishing needs DDS depth ≥ burst size (32 here) on both sides — a shallow history silently drops the tail of a burst mid-chunk.

```bash
CHUNK=8 CODEC=png bash scripts/09-bridge-run.sh 1000     # one episode
bash scripts/15-chunk-bench.sh                           # 10-episode benchmark
```

Data note: the committed `benchmarks/chunk2|4|8/` JSONs predate the latency-field rename — their `mean_step_latency_s` (~0.01 s) was measured around the env step only under the interim cadence and is superseded by `mean_round_trip_s` (observation sent → action received) in any run after commit `adc394e`.

## Real-time mode (`async_inference`, `refill_watermark`)

The classic flow is synchronous request-reply: the env asks when its FIFO runs dry, and the policy node infers **inside the subscription callback**. Two parameters loosen that coupling, in the direction Real-Time-Chunking-style serving points:

- **policy_node `async_inference:=true`** — inference moves to a worker thread. The callback only slots the newest observation; observations arriving while an inference is in flight collapse to the latest one (receding horizon), and the worker re-plans the moment it finishes. A failed inference re-arms the dedup gate, so the env's next heartbeat (~2 s) retries it instead of starving to the env's timeout. Shutdown races (kill mid-plan) exit quietly.
- **env_node `refill_watermark:=K`** (default −1 = off) — with K ≥ 0, the env publishes the latest observation **without blocking** as soon as ≤ K actions remain queued, so a re-plan is requested while earlier actions are still executing.

Measured on the chunk-8 PNG sweep (same seeds 1000–1009 / 1000–1004; this session ran on a thermally throttled GPU — compare within the table, not against the published walls above):

| config | Success | Mean max coverage | obs round-trips |
|---|---|---|---|
| sync, ask-when-dry (published sweep, cooled GPU) | 5/10 | 0.919 | 29.8 |
| async, ask-when-dry (`wm:=-1`, n=5) | 3/5 | 0.798 | = steps/8 exactly |
| async + `refill_watermark:=4` (n=10) | **0/10** | 0.646 | 224 (≈ every step) |

Reading the numbers:

- **The worker thread itself costs nothing**: per-plan latency is identical to the synchronous flow (1.6 s-class per chunk-8 denoising pass, measured back-to-back in the same session), and ask-when-dry async lands within sync chunk-8's range (3/5 vs 5/10 at these sample sizes) with round-trips exactly steps/8. Use it to keep the ROS executor responsive and for the heartbeat self-heal.
- **Proactive refill poisons this task (0/10), and the mechanism matters more than the number**: each watermark-triggered observation produces a burst that queues *behind* the actions still pending. Bursts stack several deep, every one of them conditioned on an observation captured before its queued predecessors execute, and the FIFO effectively never runs dry — so the dry-path request, the only one that re-conditions on truly fresh state, almost never fires. The failure signature is the chunk sweep's open-loop one, amplified: several episodes reach max coverage 0.98+ and cannot hold it.
- This regime is **inference-bound**: one denoising pass (1.6–2 s here) yields only 8 actions that the env consumes in milliseconds, so no watermark can hide the latency and every queued burst is stale by construction. Real-time chunking pays off when inference fits **inside** the control period (fast policies, fixed-rate control loops) — there, `async_inference` plus a small watermark is the right shape. A latest-wins mailbox (a fresh chunk supersedes the queued remainder instead of appending) is the natural next step and is not implemented yet.

```bash
CHUNK=8 WATERMARK=4 bash scripts/16-rtc-async-bench.sh                  # async + proactive refill
CHUNK=8 WATERMARK=-1 ASYNC=false bash scripts/16-rtc-async-bench.sh     # sync chunk-8 reference
```

Raw per-episode JSON: [`benchmarks/rtc_async_wm4/`](benchmarks/rtc_async_wm4/) (10 seeds) and [`benchmarks/rtc_async_wm0/`](benchmarks/rtc_async_wm0/) (5 seeds). The env stats JSON records `refill_watermark` for provenance.

## Nodes & parameters

| Node | Topic | Type | Dir |
|---|---|---|---|
| env_node | `/pusht/observation` | `pusht_ros_bridge/PushtObservation` | pub |
| env_node | `/pusht/action` | `std_msgs/Float32MultiArray` | sub |
| policy_node | `/pusht/observation` | `pusht_ros_bridge/PushtObservation` | sub |
| policy_node | `/pusht/action` | `std_msgs/Float32MultiArray` | pub |

QoS: reliable + volatile on both sides; observation depth 5, action depth **32** — the action history must be deep enough to hold a whole burst without dropping its tail.

env_node parameters: `seed` (int), `video_path` (mp4 out), `stats_path` (JSON out), `image_codec` (`jpeg`|`png`), `refill_watermark` (int, default −1 = ask-when-dry; ≥0 = receding-horizon refill, see real-time mode).
policy_node parameters: `model_path` (checkpoint dir), `action_chunk_size` (int, default 1), `async_inference` (bool, default false = synchronous request-reply).

Adapting to another policy/env: point `model_path` at a 0.6-format checkpoint, and edit the observation preprocessing in `policy_node._infer_and_publish` to match your env's obs keys. Swap `gym.make` in env_node for your environment (or a real robot driver).

## Environment

Tested with ROS 2 Jazzy on Ubuntu 24.04 (WSL2) and a Python 3.12 venv containing
`pip install "lerobot[dataset,pusht,diffusion]==0.6.1"` (torch CUDA build) with
`gym_pusht` importable in the same interpreter. The scripts default to
`/root/lerobot-venv/bin/python` and the colcon workspace `/root/ros2_ws`;
override with `PY=` / `WS=`. Migration of old-format checkpoints is covered by
`scripts/00-prepare-model.sh` (see Gotchas #2).

## Running

The companion scripts in [`scripts/`](scripts/) discover this repository from
their own location and **rsync it into the colcon workspace on every build**
(`08`), so the code that runs is always exactly what you cloned. They still
default to the workspace layout they were developed on (`/root/ros2_ws`,
`/root/lerobot-venv`, WSL2) — override `WS=` / `PY=` when porting.

```bash
# 0. one-time: prepare the migrated checkpoint (see Gotchas #2)
bash scripts/00-prepare-model.sh            # MODEL_REPO=lerobot/diffusion_pusht

# 1. one-time: build the message package (inside WSL)
bash scripts/08-bridge-build.sh

# 2. one episode (env + policy cold start, ~1-2 min on GPU)
bash scripts/09-bridge-run.sh 7             # seed

# 3. benchmarks (10 episodes each)
bash scripts/10-bridge-bench.sh             # bridge, jpeg
bash scripts/12-png-bench.sh                # bridge, png
bash scripts/11-cli-bench.sh                # lerobot CLI baseline
bash scripts/15-chunk-bench.sh              # chunked execution
bash scripts/16-rtc-async-bench.sh          # real-time mode (async + refill)
```

Why not `ros2 run`: it always launches nodes with the **system** Python, and the policy node needs `rclpy` + `lerobot` in the same (venv) interpreter. The run scripts execute both nodes with the venv Python directly, with ROS site-packages injected:

```bash
export PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/jazzy/lib:$LD_LIBRARY_PATH
```

(The alternative is creating the venv with `--system-site-packages` so ROS packages resolve inside it — see the aggregated workaround in [ros2/ros2#1094](https://github.com/ros2/ros2/issues/1094). These scripts take the explicit `PYTHONPATH` route instead, which keeps the venv hermetic.)

## Gotchas baked into the code (lerobot 0.6.1)

1. `PreTrainedConfig.from_pretrained()` does **not** backfill `cfg.pretrained_path` — without the manual backfill, the processor pipeline silently builds with empty normalization stats and the policy acts blind with no error ([#4647](https://github.com/huggingface/lerobot/issues/4647), fix PR [#4648](https://github.com/huggingface/lerobot/pull/4648)).
2. Old checkpoints must be migrated to the 0.6 processor format, and the migration tool's output directory is incomplete ([#4649](https://github.com/huggingface/lerobot/issues/4649)) — `scripts/00-prepare-model.sh` assembles the final directory.
3. `cv2.imencode` expects BGR; gym observations are RGB. Flip before encoding or the policy sees channel-swapped images.
4. Call `policy.reset()` at episode start — the diffusion policy keeps observation/action queues across episodes otherwise.
5. On load you will see `WARNING: Unexpected key(s) when loading model: [normalize_inputs.*, ...]` (8 keys). This is **expected**: the checkpoint keeps its original normalization buffers for provenance, the processor pipeline replaces them, and the policy ignores them. It is not a corrupt checkpoint.

## Limitations

- jpeg transport measurably hurts final-alignment precision (see benchmark); use `image_codec:=png` unless bandwidth demands jpeg.
- The default flow is synchronous request-reply on topics; `async_inference` moves planning off the executor and onto the freshest observation, but an action/service-based graph would be more idiomatic for tight control loops — this follows the RFC's topic-first direction.
- Single env, single policy, n=10 statistics.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Codebase comments are in English; companion workspace scripts contain some Chinese comments.
