import argparse
import json
import os
from math import comb

import numpy as np

from tau_bench.types import EnvRunResult


def display_metrics(results: list[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    # avg_reward = sum(rewards) / len(rewards)
    ddof = 1
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards, ddof=ddof)
    stde_reward = np.std(rewards, ddof=ddof) / np.sqrt(len(rewards))
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward:.2f} ± {stde_reward:.2f}")
    print(f" Standard deviation: {std_reward:.2f}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt")
    args = parser.parse_args()

    if os.path.exists(args.ckpt) and os.path.isfile(args.ckpt):
        with open(args.ckpt, "r") as f:
            data = json.load(f)

        results = []
        for record in data:
            results.append(EnvRunResult(**record))

        display_metrics(results)


if __name__ == "__main__":
    main()
