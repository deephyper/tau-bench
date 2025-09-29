# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], (
        "Only retail and airline envs are supported"
    )
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot"], (
        "Invalid agent strategy"
    )
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], (
        "Invalid user strategy"
    )

    random.seed(config.seed)

    if config.ckpt_path is None or not os.path.exists(config.ckpt_path):
        time_str = datetime.now().strftime("%m%d%H%M%S")
        ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
    else:
        ckpt_path = config.ckpt_path

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks)
        if config.end_index == -1
        else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
        )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            result: EnvRunResult | None = None
            data: list = []
            if os.path.exists(ckpt_path):
                with lock:
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)

                    # {
                    #     "task_id": 106,
                    #     "reward": 0.0,
                    #     "info": {
                    #         "error": "litellm.InternalServerError: InternalServerError: OpenAIException - litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.. Received Model Group=gpt-oss-20B-MXFP4\nAvailable Model Group Fallbacks=None",
                    #         "traceback": 'Traceback (most recent call last):\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/llms/openai/openai.py", line 736, in completion\n    raise e\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/llms/openai/openai.py", line 664, in completion\n    ) = self.make_sync_openai_chat_completion_request(\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/litellm_core_utils/logging_utils.py", line 237, in sync_wrapper\n    result = func(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/llms/openai/openai.py", line 482, in make_sync_openai_chat_completion_request\n    raise e\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/llms/openai/openai.py", line 464, in make_sync_openai_chat_completion_request\n    raw_response = openai_client.chat.completions.with_raw_response.create(\n                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/openai/_legacy_response.py", line 364, in wrapped\n    return cast(LegacyAPIResponse[R], func(*args, **kwargs))\n                                      ^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/openai/_utils/_utils.py", line 286, in wrapper\n    return func(*args, **kwargs)\n           ^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py", line 1147, in create\n    return self._post(\n           ^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/openai/_base_client.py", line 1259, in post\n    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))\n                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/openai/_base_client.py", line 1047, in request\n    raise self._make_status_error_from_response(err.response) from None\nopenai.InternalServerError: Error code: 500 - {\'error\': {\'message\': \'litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.. Received Model Group=gpt-oss-20B-MXFP4\\nAvailable Model Group Fallbacks=None\', \'type\': None, \'param\': None, \'code\': \'500\'}}\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/main.py", line 2082, in completion\n    raise e\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/main.py", line 2055, in completion\n    response = openai_chat_completions.completion(\n               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/llms/openai/openai.py", line 747, in completion\n    raise OpenAIError(\nlitellm.llms.openai.common_utils.OpenAIError: Error code: 500 - {\'error\': {\'message\': \'litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.. Received Model Group=gpt-oss-20B-MXFP4\\nAvailable Model Group Fallbacks=None\', \'type\': None, \'param\': None, \'code\': \'500\'}}\n\nDuring handling of the above exception, another exception occurred:\n\nTraceback (most recent call last):\n  File "/Users/rp5/Documents/tau-bench/tau_bench/run.py", line 78, in _run\n    res = agent.solve(\n          ^^^^^^^^^^^^\n  File "/Users/rp5/Documents/tau-bench/tau_bench/agents/tool_calling_agent.py", line 40, in solve\n    res = completion(\n          ^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/utils.py", line 1347, in wrapper\n    raise e\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/utils.py", line 1222, in wrapper\n    result = original_function(*args, **kwargs)\n             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/main.py", line 3653, in completion\n    raise exception_type(\n          ^^^^^^^^^^^^^^^\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 2273, in exception_type\n    raise e\n  File "/Users/rp5/Documents/llm-hpo/.venv/lib/python3.12/site-packages/litellm/litellm_core_utils/exception_mapping_utils.py", line 502, in exception_type\n    raise InternalServerError(\nlitellm.exceptions.InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.. Received Model Group=gpt-oss-20B-MXFP4\nAvailable Model Group Fallbacks=None\n',
                    #     },
                    #     "traj": [],
                    #     "trial": 0,
                    # }

                # Find if task is already done
                print(f"Looking for record {idx} in checkpoint")
                task_record = None
                for j in range(len(data)):
                    if data[j]["task_id"] == idx:
                        task_record = data[j]
                        break

                if task_record:
                    print(f"Found record {idx} in checkpoint")
                    result = EnvRunResult(**data[i])
                    # if "error" in task_record["info"]:
                    #     data.pop(j)

                    #     with open(ckpt_path, "w") as f:
                    #         json.dump(data, f, indent=2)
                    # else:
                    #     result = EnvRunResult(**task_record)

            if result is None:
                isolated_env = get_env(
                    config.env,
                    user_strategy=config.user_strategy,
                    user_model=config.user_model,
                    task_split=config.task_split,
                    user_provider=config.user_model_provider,
                    task_index=idx,
                )

                print(f"Running task {idx}")
                try:
                    res = agent.solve(
                        env=isolated_env,
                        task_index=idx,
                    )
                    result = EnvRunResult(
                        task_id=idx,
                        reward=res.reward,
                        info=res.info,
                        traj=res.messages,
                        trial=i,
                    )
                except Exception as e:
                    result = EnvRunResult(
                        task_id=idx,
                        reward=0.0,
                        info={"error": str(e), "traceback": traceback.format_exc()},
                        traj=[],
                        trial=i,
                    )
                print(
                    "✅" if result.reward == 1 else "❌",
                    f"task_id={idx}",
                    result.info,
                )
                print("-----")
                with lock:
                    with open(ckpt_path, "w") as f:
                        json.dump(data + [result.model_dump()], f, indent=2)
            else:
                print(f"Loaded task {idx} from checkpoint")

                print(
                    "✅" if result.reward == 1 else "❌",
                    f"task_id={idx}",
                    result.info,
                )
                print("-----")
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    return results


def agent_factory(tools_info: List[Dict[str, Any]], wiki, config: RunConfig) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            # specific inference parameters
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            repeat_penalty=config.repeat_penalty,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent

        assert config.few_shot_displays_path is not None, (
            "Few shot displays path is required for few-shot agent strategy"
        )
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
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
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
