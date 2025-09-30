import json
import re
import subprocess

import numpy as np
from deephyper.evaluator import Evaluator, RunningJob
from deephyper.evaluator.callback import TqdmCallback
from deephyper.hpo import CBO, HpProblem

DEFAULT_VALUE = {
    "temperature": 1.0,
    "top_k": 40,
    "top_p": 1.0,
    "min_p": 0.01,
    "repeat_penalty": 1.0,  # 0.0 (disabled)
    "reasoning_effort": "high",
}


def create_problem():
    # From LLAMA.cpp
    # For default values from `llama-server` we look at the CLI documentation `llama-server --help`
    # Then for the name of parameters processed by the REST API (can be different from the CLI)
    # we look at: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L273

    def get_default(name):
        return DEFAULT_VALUE.get(name)

    problem = HpProblem()

    def add_param(*args, **kwargs):
        return problem.add_hyperparameter(
            *args, **kwargs, default_value=get_default(args[1])
        )

    add_param((0.0, 5.0), "temperature")
    add_param((0.01, 1.0), "top_p")
    add_param((0.01, 1.0), "min_p")
    add_param((5, 100), "top_k")
    add_param((0.0, 2.0), "repeat_penalty")
    add_param(["low", "medium", "high"], "reasoning_effort")

    return problem


def parse_subprocess_result(result):
    """Utility to parse a result from a subprocess of the format `"DH-OUTPUT:..."`.

    Args:
        result: object returned by a subpross with ``stdout`` and ``stderr`` attributes.

    Return:
        The parsed value or raise an exception if an error happened.
    """
    stdout = result.stdout
    stderr = result.stderr
    try:
        retval_bytes = re.search(b"Average reward: (.+)\n", stdout).group(1)
    except AttributeError:
        error = stderr.decode("utf-8")
        raise RuntimeError(
            f"{error}\n\n Could not collect any result from the run_function in the main process \
            because an error happened in the subprocess."
        )
    # Finally, parse whether the return value from the user-defined function
    # is a scalar, a list, or a dictionary.
    retval = retval_bytes.replace(
        b"'", b'"'
    )  # For dictionaries, replace single quotes with double quotes!
    sol = json.loads(retval)
    return sol


def eval_benchmark(job: RunningJob):
    job_id = int(job.id.split(".")[-1])
    seed = 42 + job_id
    rng = np.random.RandomState(seed)

    # collect parameters
    temperature = job.parameters["temperature"]
    top_p = job.parameters["top_p"]
    top_k = job.parameters["top_k"]
    min_p = job.parameters["min_p"]
    repeat_penalty = job.parameters["repeat_penalty"]
    reasoning_effort = job.parameters["reasoning_effort"]

    # sample random task id
    task_split = "train"
    task_id = rng.randint(
        0,
        500,  # dev split for retail
        # 114 + 1, # test split for retail
    )
    log_dir = f"hpo/job-{job_id}"

    command = f"python run.py --agent-strategy tool-calling --env retail --model gpt-oss-20B-MXFP4 --model-provider openai --user-model gpt-oss-120B-MXFP4 --user-model-provider openai --user-strategy llm --max-concurrency 1 --temperature {temperature} --min_p {min_p} --top_k {top_k} --top_p {top_p} --repeat_penalty {repeat_penalty} --reasoning_effort {reasoning_effort} --task-ids {task_id} --task-split {task_split} --log-dir {log_dir}"

    try:
        completed_process = subprocess.run(command.split(), capture_output=True)
    except Exception:
        objective = "F_subprocess_run"
    else:
        try:
            objective = parse_subprocess_result(completed_process)
        except Exception:
            objective = "F_parse"

    return {
        "objective": objective,
        "metadata": {
            "task_id": int(task_id),
        },
    }


def main():
    problem = create_problem()
    search = CBO(
        problem,
        initial_point_generator="lhs",
        n_initial_points=100,
        surrogate_model="ET",
        surrogate_model_kwargs={
            "n_estimators": 100,
            "min_samples_leaf": 1,
            "min_samples_split": 8,
            "bootstrap": False,
            "max_features": "sqrt",
        },
        acq_func="UCBd",
        acq_func_kwargs={
            "kappa": 1.96,
            "scheduler": {
                "type": "periodic-exp-decay",
                "period": 100,
                "kappa_final": 0.001,
            },
        },
        acq_optimizer="sampling",
        acq_optimizer_kwargs={
            "acq_optimizer_freq": 10,
            "outliers_iqr_factor": float("inf"),
        },
        solution_selection="argmax_est",
    )

    evaluator = Evaluator.create(
        eval_benchmark,
        method="thread",
        method_kwargs={"num_workers": 2, "callbacks": [TqdmCallback()]},
    )

    search.search(evaluator, max_evals=1000)


if __name__ == "__main__":
    main()
