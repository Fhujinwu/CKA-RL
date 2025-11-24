from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import pandas as pd
import argparse
import pathlib
import os, sys

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="Debug", type=str,
        help="directory where the tensorboard data is stored")
    parser.add_argument("--no-cache", default=False, action="store_true",
        help="wheter to disable the cache option. If not provided and `--save-dir` exists, skips processing tensorboard files")
    # fmt: on
    return parser.parse_args()


def parse_metadata(ea):
    md = ea.Tensors("hyperparameters/text_summary")[0]
    md_bytes = md.tensor_proto.SerializeToString()

    # remove first non-ascii characters and parse
    start = md_bytes.index(b"|")
    md_str = md_bytes[start:].decode("ascii")

    md = {}
    for row in md_str.split("\n")[2:]:
        s = row.split("|")[1:-1]
        k = s[0]
        if s[1].isdigit():
            v = int(s[1])
        elif s[1].replace(".", "").isdigit():
            v = float(s[1])
        elif s[1] == "True" or s[1] == "False":
            v = s[1] == "True"
        else:
            v = s[1]
        md[k] = v
    return md


def parse_tensorboard(path, scalars, single_pts=[]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()

    # make sure the scalars are in the event accumulator tags
    if sum([s not in ea.Tags()["scalars"] for s in scalars]) > 0:
        print(f"** Scalar not found. Skipping file {path}")
        return None

    md = parse_metadata(ea)

    for name in single_pts:
        if name in ea.Tags()["scalars"]:
            md[name] = pd.DataFrame(ea.Scalars(name))["value"][0]

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}, md


if __name__ == "__main__":
    sys.path.append("../../")

    args = parse_args()

    # hardcoded settings
    scalar = "charts/success"
    final_success = "charts/test_success"

    # Extract data from tensorboard results to an actually useful CSV
    runs_dir = f"./runs/{args.tag}"
    save_csv = f"./data/{args.tag}/extract_results.csv"
    exists = os.path.exists(save_csv)
    if args.no_cache or (not exists and not args.no_cache):
        dfs = []
        for path in tqdm(list(pathlib.Path(runs_dir).rglob("*events.out*"))):
            # print("*** Processing ", path)
            res = parse_tensorboard(str(path), [scalar], [final_success])
            if res is not None:
                dic, md = res
            else:
                print("No data. Skipping...")
                continue

            df = dic[scalar]
            df = df[["step", "value"]]
            df["seed"] = md["seed"]
            df["task_id"] = md["task_id"]
            df["model_type"] = md["model_type"]

            if final_success in md:
                df[final_success] = md[final_success]

            dfs.append(df)
        df = pd.concat(dfs)
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)
    else:
        print(f"** Loading cached CSV file {save_csv}")