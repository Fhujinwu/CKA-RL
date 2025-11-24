import subprocess
import argparse
import random
from task_utils import TASKS
# from icecream import ic
from loguru import logger
    
method_choices = ["Baseline",         # F1
                  "Finetune",         # FN
                  "CompoNet",         # CompoNet
                  "PackNet",          # PackNet
                  "ProgNet",          # ProgNet
                  "CKA-RL",           # CKA-RL
                  "MaskNet",          # MaskNet
                  "CbpNet",           # CbpNet
                  "CReLUs",           # Concatenated ReLUs CoLLAs 2023 
                  ]

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

    parser.add_argument("--method_type", type=str, choices=method_choices, required=True)
    parser.add_argument("--env", type=str, choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/Freeway-v5")
    parser.add_argument("--seed", type=int, required=False, default=42)

    parser.add_argument("--first-mode", type=int, required=True)
    parser.add_argument("--last-mode", type=int, required=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--total_timesteps", type=int, default=int(1e6))
    parser.add_argument("--alpha_factor", type=float, default=None)
    parser.add_argument("--alpha_learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--delta_theta_mode", type=str, default="T", choices=["T","TAT"]) # T = theta, TAT = theta + alpha*tau
    parser.add_argument("--fuse_actor", type=str2bool, default=True)
    parser.add_argument("--reset_actor", type=str2bool, default=True) # reset actor
    parser.add_argument("--global_alpha", type=str2bool, default=True) # wehter to use global alpha for whole agent
    parser.add_argument("--alpha_init", type=str, default="Randn") 
    parser.add_argument("--alpha_major", type=float, default=0.6) 
    parser.add_argument("--pool_size", type=int, default=5) 
    
    return parser.parse_args()


args = parse_args()
logger.info(f"experiments args : {args}")
modes = TASKS[args.env]

first_mode = args.first_mode
last_mode = args.last_mode
debug = args.debug
method_type = args.method_type
logger.debug(f"Experiment Tag: {args.tag}")

seed = random.randint(0, 1e6) if args.seed is None else args.seed

env_name = args.env.split("/")[1].split("-")[0] # e.g. ALE/Freeway-v5 -> Freeway
run_name = (
    lambda task_id: f"{env_name}_{task_id}_{args.method_type}_{args.seed}" # e.g. Freeway_1_FN
)

first_idx = modes.index(first_mode)
last_idx = modes.index(last_mode)

for i, task_id in enumerate(modes[first_idx:last_idx+1]):
    # params
    save_dir = f"agents/{env_name}/{args.tag}"
    params = f"--method-type={method_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id}"
    params += f" --task_id={task_id}"
    params += f" --tag={args.tag}"
    params += f" --total_timesteps={args.total_timesteps}"
    params += f" --delta_theta_mode={args.delta_theta_mode}"
    params += (f" --fuse_actor" if args.fuse_actor else f" --no-fuse_actor")
    params += (f" --reset_actor" if args.reset_actor else f" --no-reset_actor")
    params += (f" --global_alpha" if args.global_alpha else f" --no-global_alpha")
    params += f" --alpha_init={args.alpha_init}" 
    params += f" --alpha_major={args.alpha_major}" 
    params += f" --pool_size={args.pool_size}" 
    
    if args.alpha_factor is not None:
        params += f" --alpha_factor={args.alpha_factor}"
    params += f" --alpha_learning_rate={args.alpha_learning_rate}"
        
    # debug mode
    params += (" --track" if not debug else " --no-track")
    params += (" --debug" if debug else " --no-debug")
    if debug:
        logger.debug(f"Running experiment within debugging mode")
        params += f" --total-timesteps=3000"
        
    # method specific CLI arguments
    if args.method_type == "CompoNet":
        params += " --componet-finetune-encoder"
    if args.method_type in ["PackNet", "MaskNet"]:
        params += f" --total-task-num={len(modes)}"

    if i > 0:
        # multiple previous modules
        if args.method_type in ["CompoNet", "ProgNet", "CKA-RL"]:
            params += " --prev-units"
            logger.info(f"Method {args.method_type} need prevs-units, adding prevs-units")
            for j in range(len(modes[: modes.index(task_id)])):
                params += f" {save_dir}/{run_name(j)}"
        # single previous module
        elif args.method_type in ["Finetune", "PackNet", "MaskNet", "CbpNet", "CReLUs"]:
            params += f" --prev-units {save_dir}/{run_name(i-1)}"
            
    # Launch experiment
    cmd = f"python run_ppo.py {params}"
    logger.info(f"Running experiment script: {cmd}")
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        logger.error(f"Process {cmd} \n returned code {res.returncode}. Stopping on error.")
        quit(1)
