#!/bin/bash
# CLI 直连 10 回合基准：同模型同 seed 集（1000-1009）
set -e
source /root/lerobot-venv/bin/activate
export SDL_VIDEODRIVER=dummy
export HF_HOME=/root/hf-cache

lerobot-eval --env.type=pusht --policy.path=/root/diffusion_pusht_migrated \
    --eval.n_episodes=10 --seed 1000 --eval.use_async_envs false \
    --output_dir /root/cli_bench --job_name cli_bench 2>&1 | tail -4

echo "=== 汇总 ==="
python - <<'EOF'
import json
info = json.load(open('/root/cli_bench/eval_info.json'))
per = info.get('per_task', [{}])[0].get('metrics', {}) if 'per_task' in info else {}
print(json.dumps({k: v for k, v in info.items() if k in
                  ('avg_sum_reward', 'avg_max_reward', 'pc_success', 'n_episodes')},
                 indent=1))
succ = per.get('successes', [])
if succ:
    print('successes:', succ)
    sr = per.get('sum_rewards', [])
    print(f'sum_reward: mean={sum(sr)/len(sr):.1f} min={min(sr)} max={max(sr)}')
EOF
echo CLI_BENCH_DONE
