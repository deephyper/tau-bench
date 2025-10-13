import argparse
import json
import pandas as pd

from deephyper.analysis.hpo import read_results_from_csv, filter_failed_objectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--job-id", type=int, default=None)
    parser.add_argument("--n-last", type=int, default=None)
    args = parser.parse_args()

    results: pd.DataFrame = read_results_from_csv(args.path)
    results, _ = filter_failed_objectives(results)

    if args.job_id is None or args.n_last is None:
        if args.job_id is not None:
            if args.job_id < 0:
                job_id = results["job_id"].sort_values().iloc[args.job_id]
            else:
                job_id = args.job_id

        if args.job_id is None:
            job_id = results["job_id"].max()

        config = results[results["job_id"] == job_id].iloc[0].to_dict()

    if args.n_last is not None:
        n_last = args.n_last

        idx = results.iloc[-n_last:]["sol.objective"].argmax()
        config = results.iloc[-n_last:].iloc[idx].to_dict()

    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
