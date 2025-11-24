import pandas as pd
import numpy as np
import pathlib
import argparse
from tabulate import tabulate
import sys, os

sys.path.append("../../")

SETTINGS = {
    "SpaceInvaders": dict(
        ma_w_1=10,
        num_pts_sc=100,
        sc_percent=1.0,
        chunk_avg_w=30,
        ma_w_extra=30,
        ma_std_extra=10,
    ),
    "Freeway": dict(
        ma_w_1=10,
        num_pts_sc=100,
        sc_percent=1.0,
        chunk_avg_w=30,
        ma_w_extra=10,
        ma_std_extra=None,
    ),
}

METHOD_NAMES = {
    "Baseline": "Baseline",
    "Finetune": "Finetune",
    "CompoNet": "CompoNet",
    "ProgNet": "ProgressiveNet",
    "PackNet": "PackNet",
    "CKA": "CKA-RL",
    "CReLUs": "CReLUs",
    "MaskNet": "MaskNet",
    "CbpNet": "CbpNet",
}

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/Freeway",
        help="path to the directory where the CSV of each task is stored")
    # fmt: on
    return parser.parse_args()


def remove_nan(x, y):
    no_nan = ~np.isnan(y)
    return x[no_nan], y[no_nan]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def chunk_average(x, w):
    split = np.array_split(x, x.shape[0] // w)
    x_avg = np.array([chunk.mean() for chunk in split])
    x_std = np.array([chunk.std() for chunk in split])
    return x_avg, x_std


def compute_success(
    df,
    ma_w_1=10,
    num_pts_sc=100,
    sc_percent=0.8,
    chunk_avg_w=30,
    ma_w_extra=30,
    ma_std_extra=10,
):
    data_cols = df.columns[df.columns.str.endswith("episodic_return")]
    methods = [col.split("-")[0] for col in data_cols]

    rets = []
    returns = {}
    for method, col in zip(methods, data_cols):
        x, y = df["global_step"].values, df[col].values
        x, y = remove_nan(x, y)

        x = moving_average(x, w=ma_w_1)
        y = moving_average(y, w=ma_w_1)
        returns[method] = (x, y)

        rets.append(y[:-num_pts_sc].mean())

    success_score = sc_percent * np.mean(rets)

    data = {}
    for method in methods:
        x, y = returns[method]
        y = y >= success_score

        x, _ = chunk_average(x, w=chunk_avg_w)
        y, y_std = chunk_average(y, w=chunk_avg_w)

        if ma_w_extra is not None:
            x = moving_average(x, w=ma_w_extra)
            y = moving_average(y, w=ma_w_extra)
            y_std = moving_average(y_std, w=ma_w_extra)

        if ma_std_extra is not None:
            x_std = moving_average(x, w=10)
            y_min = moving_average(np.maximum(y - y_std, 0), w=10)
            y_max = moving_average(np.minimum(y + y_std, 1), w=10)
        else:
            x_std = x
            y_min = np.maximum(y - y_std, 0)
            y_max = np.minimum(y + y_std, 1)

        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
        y_std = np.insert(y_std, 0, 0.0)
        y_min = np.insert(y_min, 0, 0.0)
        y_max = np.insert(y_max, 0, 0.0)
        x_std = np.insert(x_std, 0, 0.0)

        d = {}
        d["global_step"] = x
        d["success"] = y
        d["std_high"] = y_max
        d["std_low"] = y_min
        d["std_x"] = x_std
        d["final_success"] = np.mean(y[-100:])
        d["final_success_std"] = np.std(y[-100:])

        data[method] = d

    return data, success_score

def compute_forward_transfer(data):
    baseline_method = "Baseline"
    methods = list(METHOD_NAMES.keys())

    ft_data = {}
    for task_id in data.keys():
        ft_data[task_id] = {}

        # get the baseline's data
        task_data = data[task_id]
        x_baseline = task_data[baseline_method]["global_step"]
        y_baseline = task_data[baseline_method]["success"]
        baseline_area_down = np.trapz(x=x_baseline, y=y_baseline)
        baseline_area_up = np.max(x_baseline) - baseline_area_down

        for method in methods:
            x_method = task_data[method]["global_step"]
            y_method = task_data[method]["success"]
            y_baseline = task_data[baseline_method]["success"]

            # get a common X axis
            x = []
            mi, bi = 0, 0
            while mi < len(x_method) and bi < len(x_baseline):
                if x_method[mi] < x_baseline[bi]:
                    x.append(x_method[mi])
                    mi += 1
                else:
                    x.append(x_baseline[bi])
                    bi += 1
            x = np.array(x)
            y_baseline = np.interp(x, x_baseline, y_baseline)
            y_method = np.interp(x, x_method, y_method)

            # compute the actual FT
            up_idx = y_method > y_baseline
            down_idx = y_method < y_baseline

            area_up = np.trapz(y=y_method[up_idx], x=x[up_idx]) - np.trapz(
                y=y_baseline[up_idx], x=x[up_idx]
            )
            area_down = np.trapz(y=y_baseline[down_idx], x=x[down_idx]) - np.trapz(
                y=y_method[down_idx], x=x[down_idx]
            )
            ft = (area_up - area_down) / baseline_area_up

            ft_data[task_id][method] = ft

    #
    # Printing the results in a pretty table
    #
    methods = list(METHOD_NAMES.keys())
    table = []
    for task_id in sorted(ft_data.keys()):
        row = [task_id]
        for i, method in enumerate(methods):
            val = round(ft_data[task_id][method], 4)
            row.append(val)
        table.append(row)
    table.append([None] * len(method))

    # compute the average and std FT of every method
    avgs = []
    for method in methods:
        method_avg = []
        for task_id in sorted(ft_data.keys())[
            1:
        ]:  # ignore the first task to compute the avg. ft
            method_avg.append(ft_data[task_id][method])
        mean = round(np.mean(method_avg), 4)
        std = round(np.std(method_avg), 4)
        avgs.append(f"{mean} ({std})")
    table.append(["Avg."] + avgs)

    print("\n\n----- FORWARD TRANSFER -----\n")
    print(
        tabulate(
            table,
            headers=["Task ID"] + [METHOD_NAMES[m] for m in methods],
            tablefmt="rounded_outline",
        )
    )

    return ft_data

def compute_final_performance(data):
    methods = list(METHOD_NAMES.keys())
    table = []
    for task_id in sorted(data.keys()):
        row = [task_id]
        for i, method in enumerate(methods):
            val = round(data[task_id][method]["final_success"], 4)
            row.append(val)
        table.append(row)
    table.append([None] * len(method))

    avgs = []
    for j in range(1, len(table[0])):  # skip task id's column
        m = []
        for i in range(len(table) - 1):  # skip Nones row
            m.append(table[i][j])
        mean = round(np.mean(m), 4)
        std = round(np.std(m), 4)
        avgs.append(f"{mean} ({std})")

    table.append(["Avg."] + avgs)

    print("\n\n----- PERFORMANCE -----\n")
    print(
        tabulate(
            table,
            headers=["Task ID"] + [METHOD_NAMES[m] for m in methods],
            tablefmt="rounded_outline",
        )
    )

if __name__ == "__main__":
    args = parse_args()

    env = os.path.basename(args.data_dir)

    data = {}
    scores = {}
    for path in pathlib.Path(args.data_dir).glob("*.csv"):
        task_id = int(str(path)[:-4].split("_")[-1])  # obtain task ID from the path

        df = pd.read_csv(path)

        cfg = dict()
        for k in SETTINGS.keys():
            if k in str(path):
                cfg = SETTINGS[k]
                break
        print(task_id)
        data_task, success_score = compute_success(df, **cfg)
        data[task_id] = data_task
        scores[task_id] = success_score

    print("\n** Success scores used:")
    [print(round(scores[t], 4), end=" ") for t in sorted(scores.keys())]
    print()

    ft_data = compute_forward_transfer(data)
    compute_final_performance(data)
