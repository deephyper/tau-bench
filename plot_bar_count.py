import argparse
import json

import matplotlib.pyplot as plt

from deephyper.analysis.hpo import read_results_from_csv, filter_failed_objectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--size-bin", type=int, default=None)
    args = parser.parse_args()

    results = read_results_from_csv(args.path)
    results, _ = filter_failed_objectives(results)

    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    plt.style.use("bmh")
    gr = 1.618
    width_plot = 8
    height_plot = width_plot / gr

    if args.size_bin is None:
        num_bins = args.num_bins
        size_bin = len(results) // num_bins
    else:
        size_bin = args.size_bin
        num_bins = len(results) // size_bin + int(len(results) % size_bin > 0)
    print(f"{num_bins=}, {size_bin=}")
    counts_0, counts_1 = [], []
    for i in range(num_bins):
        counts = (
            results["objective"]
            .iloc[i * size_bin : (i + 1) * size_bin]
            .astype(int)
            .value_counts()
            .to_dict()
        )
        counts_0.append(counts.get(0, 0))
        counts_1.append(counts.get(1, 0))
    x = [v for v in range(len(counts_0))]

    fig, ax = plt.subplots(figsize=(width_plot, height_plot))

    ax.bar(x, counts_0, align="edge", width=0.99, label="0")
    ax.bar(x, height=counts_1, bottom=counts_0, align="edge", width=0.99, label="1")
    ax.legend(title="Observation")
    ax.set_ylabel("Count")
    ax.set_xlabel("Jobs Done")
    ax.set_xticks(ticks=x, labels=[size_bin * i for i in x])
    plt.show()


if __name__ == "__main__":
    main()
