import pandas as pd
import numpy as np
import os, sys
import argparse
from tabulate import tabulate

NUM_TASKS = 20

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs_all", type=str,
        help="directory where the tensorboard data is stored")
    parser.add_argument("--no-cache", default=False, action="store_true",
        help="wheter to disable the cache option. If not provided and `--save-dir` exists, skips processing tensorboard files")
    parser.add_argument("--save-csv", default="data/agg_results.csv", type=str,
        help="filename of the CSV to store the processed tensorboard results. Once processed, can be used as cache.")
    parser.add_argument("--tag", default="main", type=str)
    parser.add_argument("--smoothing-window", type=int, default=100)
    # fmt: on
    return parser.parse_args()


def smooth_avg(df, xkey, ykey, w=20):
    g = df.groupby(xkey)[ykey]
    mean = g.mean().rolling(window=w).mean()
    std = g.std().rolling(window=w).mean()

    y = mean.values
    y_std = std.values

    x = mean.reset_index()[xkey].values

    return x, y, y_std


def areas_up_down(method_x, method_y, baseline_x, baseline_y):
    up_idx = method_y > baseline_y
    down_idx = method_y < baseline_y

    assert (
        method_x == baseline_x
    ).all(), "The X axis of the baseline and method must be equal."
    x = method_x

    area_up = np.trapz(y=method_y[up_idx], x=x[up_idx]) - np.trapz(
        y=baseline_y[up_idx], x=x[up_idx]
    )
    area_down = np.trapz(y=baseline_y[down_idx], x=x[down_idx]) - np.trapz(
        y=method_y[down_idx], x=x[down_idx]
    )
    return area_up, area_down


def remove_nan(x, y):
    no_nan = ~np.isnan(y)
    return x[no_nan], y[no_nan]


def compute_forward_transfer(df, methods, smoothing_window):
    methods = methods.copy()
    methods.remove("simple")

    table = []
    results = {}
    for task_id in range(NUM_TASKS):
        # do not compute forward transfer for the first task
        if task_id == 0:
            table.append([0] + [None] * len(methods))
            continue

        baseline = df[
            (df["model_type"] == "simple") & (df["task_id"] == (task_id % 10))
        ]

        # get the curve of the `simple` method
        x_baseline, y_baseline, _ = smooth_avg(
            baseline, xkey="step", ykey="value", w=smoothing_window
        )
        x_baseline, y_baseline = remove_nan(x_baseline, y_baseline)

        baseline_area_down = np.trapz(y=y_baseline, x=x_baseline)
        baseline_area_up = np.max(x_baseline) - baseline_area_down

        table_row = [task_id]
        for j, name in enumerate(methods):
            method = df[(df["model_type"] == name) & (df["task_id"] == task_id)]
            x_method, y_method, _ = smooth_avg(
                method, xkey="step", ykey="value", w=smoothing_window
            )
            x_method, y_method = remove_nan(x_method, y_method)

            # this can happen if a method hasn't the results for all tasks
            if len(x_baseline) > len(x_method):
                table_row.append(None)
                continue

            area_up, area_down = areas_up_down(
                x_method,
                y_method,
                x_baseline,
                y_baseline,
            )
            if task_id == 18 and (name in ["componet"] or "fuse" in name):
                print(f"Method {name} task {task_id} up: {area_up} down: {area_down}")
                print(baseline_area_up)
            ft = (area_up - area_down) / baseline_area_up
            table_row.append(round(ft, 2))

            if methods[j] not in results.keys():
                results[methods[j]] = []
            results[methods[j]].append(ft)

        table.append(table_row)

    table.append([None] * len(table_row))

    row = ["Avg."]
    for i in range(len(methods)):
        vals = []
        for r in table[1:]:
            v = r[i + 1]
            if v is not None:
                vals.append(v)
        mean = np.mean(vals)
        std = np.std(vals)
        row.append((round(mean, 4), round(std, 4)))
    table.append(row)

    print("\n== FORWARD TRANSFER ==")
    print(
        tabulate(
            table,
            headers=["Task ID"] + methods,
            tablefmt="rounded_outline",
        )
    )
    print("\n")
    return results


def compute_performance(df, methods):
    table = []
    avgs = [[] for _ in range(len(methods))]
    results = {}
    eval_steps = df["step"].unique()[-10]
    for i in range(NUM_TASKS):
        row = [i]
        for j, m in enumerate(methods):
            task_id = i if m != "simple" else i % 10

            method = "simple" if task_id == 0 and m in ["componet", "finetune", "fusenet_merge"] else m

            s = df[
                (df["task_id"] == task_id)
                & (df["model_type"] == method)
                & (df["step"] >= eval_steps)
            ]["value"].values

            if len(s) == 0:
                s = np.array([np.nan])

            if methods[j] not in results.keys():
                results[methods[j]] = []
            results[methods[j]].append((s.mean(), s.std()))

            avg = round(s.mean(), 2)
            std = round(s.std(), 2)

            row.append((avg, std))
            if not np.isnan(s).any() and i > 0:
                avgs[j] += list(s)

        table.append(row)

    avgs = [(round(np.mean(v), 6), round(np.std(v), 2)) for v in avgs]
    table.append([None] * len(row))
    table.append(["Avg."] + avgs)

    for i in range(len(table)):
        for j in range(len(table[0])):
            e = table[i][j]
            if type(e) != tuple:
                continue
            if np.isnan(e[0]):
                table[i][j] = None
    print("\n== PERFORMANCE ==")
    print(
        tabulate(
            table,
            headers=["Task ID"] + methods,
            tablefmt="rounded_outline",
        )
    )
    print("\n")

    return results


if __name__ == "__main__":
    sys.path.append("../../")

    args = parse_args()
    # hardcoded settings
    methods = ["simple", "componet", "finetune", "prognet", "packnet", "cka-rl", "masknet", "cbpnet", "crelus"]

    # Extract data from tensorboard results to an actually useful CSV
    args.save_csv = f"data/{args.tag}/extract_results.csv"
    exists = os.path.exists(args.save_csv)
    print(f"\n\nReloading cache data from: {args.save_csv}")
    df = pd.read_csv(args.save_csv)

    # Compute performance and forward transfer
    data_perf = compute_performance(df, methods)
    ft_data = compute_forward_transfer(df, methods, args.smoothing_window)
