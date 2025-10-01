import argparse
import json

from deephyper.analysis.hpo import read_results_from_csv, filter_failed_objectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--job-id", type=int, default=None)
    args = parser.parse_args()

    results = read_results_from_csv(args.path)
    results, _ = filter_failed_objectives(results)

    if args.job_id is None:
        job_id = results["job_id"].max()
    else:
        job_id = job_id

    config = results[results["job_id"] == job_id].iloc[0].to_dict()
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
