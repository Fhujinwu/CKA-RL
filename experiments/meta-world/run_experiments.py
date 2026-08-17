import subprocess
import argparse
import random
from tasks import tasks

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=[
            "simple",
            "componet",
            "finetune",
            "prognet",
            "packnet",
            "cka-rl",
            "masknet",
            "cbpnet",
            "crelus"
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-run", default=False, action="store_true")
    parser.add_argument("--start-mode", type=int, default=0)
    parser.add_argument("--tag", type=str, default="Debug")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--pool_size", default=20)
    parser.add_argument("--encoder_from_base", action="store_true")
    parser.add_argument("--fuse_heads", action="store_true")
    parser.add_argument("--delta_theta_mode", type=str, default="T")
    parser.add_argument("--alpha_init", type=str, default="Randn")
    parser.add_argument("--alpha_major", type=float, default=0.6)
    parser.add_argument("--alpha_factor", type=float, default=1e-3)
    parser.add_argument("--fix_alpha", action="store_true")
    parser.add_argument("--num-tasks", type=int, default=0,
        help="run num-tasks tasks, 0 means all tasks")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="timesteps for each task, default 1e6")
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3,
        help="learning rate for alpha, default 1e-3")
    parser.add_argument("--anneal_alpha_lr", type=str2bool, default=True,
        help="whether to anneal alpha learning rate, default True")
    return parser.parse_args()


args = parse_args()

modes = list(range(20)) if args.algorithm != "simple" else list(range(10))
# args.start_mode = 3
if args.algorithm not in ["simple", "packnet", "prognet", "cka-rl", "masknet", "cbpnet", "crelus"] and args.start_mode == 0:
    start_mode = 1
else:
    start_mode = args.start_mode

run_name = (
    lambda task_id: f"task_{task_id}__{args.algorithm if task_id > 0 or args.algorithm in ['packnet', 'prognet', 'cka-rl', 'masknet', 'cbpnet', 'crelus'] else 'simple'}__run_sac__{args.seed}"
)

first_idx = modes.index(start_mode)
run_modes = modes[first_idx:]
if args.num_tasks > 0:
    run_modes = run_modes[:args.num_tasks]

for i, task_id in enumerate(run_modes):
    params = f"--model-type={args.algorithm} --task-id={task_id} --seed={args.seed} --tag={args.tag}"
    if args.debug:
        params += " --total-timesteps=50"
        params += " --learning_starts=5"
    else:
        params += f" --total-timesteps={args.total_timesteps}"
    if args.encoder_from_base:
        params += " --encoder-from-base"
    else:
        params += " --no-encoder-from-base"
    
    save_dir = f"agents/{args.tag}"
    params += f" --save-dir={save_dir}"
    params += f" --pool_size={args.pool_size}"
    if args.fuse_heads:
        params += " --fuse-heads"
    params += f" --delta-theta-mode={args.delta_theta_mode}"
    params += f" --alpha-init={args.alpha_init}"
    params += f" --alpha-major={args.alpha_major}"
    params += f" --alpha-factor={args.alpha_factor}"
    if args.fix_alpha:
        params += " --fix-alpha"
    params += f" --alpha-learning-rate={args.alpha_learning_rate}"
    if args.anneal_alpha_lr:
        params += " --anneal-alpha-lr"
    else:
        params += " --no-anneal-alpha-lr"

    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prognet", "cka-rl"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]:
                params += f" {save_dir}/{run_name(i)}"
        # single previous module
        elif args.algorithm in ["finetune", "packnet", "masknet", "cbpnet", "crelus"]:
            params += f" --prev-units {save_dir}/{run_name(task_id-1)}"

    # Launch experiment
    cmd = f"python3 run_sac.py {params}"
    print(cmd)

    if not args.no_run:
        res = subprocess.run(cmd.split(" "))
        if res.returncode != 0:
            print(f"*** Process returned code {res.returncode}. Stopping on error.")
            quit(1)
