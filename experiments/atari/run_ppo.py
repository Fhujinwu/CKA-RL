# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
import sys
from typing import Literal, Tuple, Optional
import pathlib
from loguru import logger
from tqdm import tqdm
from models.cbp_modules import GnT
from utils.AdamGnT import AdamGnT

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    CkaRlAgent,
    CnnMaskAgent,
    CnnCbpAgent,
    CReLUsAgent,
)


@dataclass
class Args:
    # Model type
    method_type: str = "Baseline"
    """The name of the model to use as agent."""
    dino_size: Literal["s", "b", "l", "g"] = "s"
    """Size of the dino model (only needed when using dino)"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Only used when employing a CompoNet or cnn-simple-ft (finetune) agent"""
    mode: int = None
    """Playing mode for the Atari game. The default mode is used if not provided"""
    componet_finetune_encoder: bool = False
    """Whether to train the CompoNet's encoder from scratch of finetune it from the encoder of the previous task"""
    total_task_num: Optional[int] = None
    """Total number of tasks, required when using PackNet"""
    prevs_to_noise: Optional[int] = 0
    """Number of previous policies to set to randomly selected distributions, only valid when method_type is `CompoNet`"""

    # Experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Atari-PPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/Freeway-v5"
    """the id of the environment"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    fuse_lr_scale: float = 100.0
    """scale the learning rate of the alpha parameter"""
    
    debug: bool = False
    """log level"""
    tag: str = "Debug"
    """experiment tag"""
    
    alpha_factor: float = 1e-2
    """fuse net's alpha initialization factor 1 * alpha_factor"""
    fix_alpha: bool = False
    """fuse net's alpha would be fix to constant"""
    alpha_learning_rate: float = 2.5e-4
    """the learning rate of alpha optimizer"""
    delta_theta_mode: str = "T" # T or TAT
    """the mode to cacluate delta theta"""
    fuse_encoder: bool = False # True or False
    """whether to fuse encoder"""
    fuse_actor: bool = True # True or False
    """whether to fuse actor"""
    reset_actor: bool = True # True or False
    """whether to reset actor"""
    global_alpha: bool = True # True or False
    """whether to use global alpha for whole agent"""
    alpha_init: str = "Randn" # "Randn" "Major" "Uniform"
    """how to init alpha in CKA-RL"""
    alpha_major: float = 0.6 
    """init major""" # Major alpha init, theta_{i-1} will be init to log(major) + C, others will be uniform
    pool_size: int = 3
    """pool size for knowledge vector pool in CKA-RL"""

    task_id: int = 0
    """task id for the current task"""

def make_env(env_id, idx, capture_video, run_name, mode=None):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)

        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)        
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

    
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.debug is False:
        logger.remove() 
        handler_id = logger.add(sys.stderr, level="INFO") 
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # m = f"{args.mode}" if args.mode is not None else ""
    m = f"{args.task_id}" if args.mode is not None else ""
    env_name = args.env_id.split("/")[1].split("-")[0] # e.g. ALE/Freeway-v5 -> Freeway
    run_name = f"{env_name}_{m}_{args.method_type}_{args.seed}"
    ao_exist = False # has alpha_optimizer if True
    
    logger.info(f"Run's name: {run_name}")

    logs = {"global_step": [0], "episodic_return": [0]}
    # logger.info(f"Tensorboard writing to runs/{args.tag}/{run_name}")
    writer = SummaryWriter(f"runs/{args.tag}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, mode=args.mode)
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    logger.info(f"Method: {args.method_type}")
    if args.method_type == "Baseline":
        agent = CnnSimpleAgent(envs).to(device)
    elif args.method_type == "Finetune":
        if len(args.prev_units) > 0:
            agent = CnnSimpleAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CnnSimpleAgent(envs).to(device)
    elif args.method_type == "CompoNet":
        agent = CnnCompoNetAgent(
            envs,
            prevs_paths=args.prev_units,
            finetune_encoder=args.componet_finetune_encoder,
            map_location=device,
        ).to(device)
    elif args.method_type == "ProgNet":
        agent = ProgressiveNetAgent(
            envs, prevs_paths=args.prev_units, map_location=device
        ).to(device)
    elif args.method_type == "PackNet":
        # retraining in 20% of the total timesteps
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)

        if args.total_task_num is None:
            print("CLI argument `total_task_num` is required when using PackNet.")
            quit(1)

        if len(args.prev_units) == 0:
            agent = PackNetAgent(
                envs,
                task_id=(args.mode + 1),
                is_first_task=True,
                total_task_num=args.total_task_num,
            ).to(device)
        else:
            agent = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.mode + 1,
                restart_actor_critic=True,
                freeze_bias=True,
            ).to(device)
    elif args.method_type == "CKA-RL":
        base_dir = args.prev_units[0] if len(args.prev_units) > 0 else None
        latest_dir = args.prev_units[-1] if len(args.prev_units) > 0 else None
        agent = CkaRlAgent(envs, 
                             base_dir=base_dir, 
                             latest_dir=latest_dir,
                             alpha_factor=args.alpha_factor,
                             fix_alpha=args.fix_alpha,
                             delta_theta_mode=args.delta_theta_mode,
                             fuse_encoder=args.fuse_encoder,
                             fuse_actor=args.fuse_actor,
                             reset_actor=args.reset_actor,
                             global_alpha=args.global_alpha,
                             alpha_init=args.alpha_init,
                             alpha_major=args.alpha_major,
                             pool_size=args.pool_size,
                             map_location=device).to(device)
        agent.log_alphas()
    elif args.method_type == "MaskNet":
        logger.info(f"num_task: {args.total_task_num}")
        logger.info(f"task: {args.mode}")
        if len(args.prev_units) > 0:
            logger.info(f"loading from {args.prev_units[0]}")
            agent = CnnMaskAgent.load(args.prev_units[0], envs, num_tasks=args.total_task_num, load_critic=False, reset_actor=False).to(device)
            agent.set_task(args.mode, new_task=True)
        else:
            agent = CnnMaskAgent(envs, num_tasks=args.total_task_num).to(device)
            agent.set_task(args.mode, new_task=False)
    elif args.method_type == "CbpNet":
        if len(args.prev_units) > 0:
            agent = CnnCbpAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CnnCbpAgent(envs).to(device)
    elif args.method_type == "CReLUs":
        if len(args.prev_units) > 0:
            agent = CReLUsAgent.load(
                args.prev_units[0], envs, load_critic=False, reset_actor=False
            ).to(device)
        else:
            agent = CReLUsAgent(envs).to(device)
    else:
        logger.error(f"Method type {args.method_type} is not valid.")
        quit(1)
        
    # print(agent)
    trainable_params = [param for name, param in agent.named_parameters() if param.requires_grad and not "alpha" in name]

    # for name, param in agent.named_parameters():
    #     if id(param) in [id(p) for p in trainable_params]:
    #         print(f"Trainable Parameter: {name}")
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, eps=1e-5)
    if args.method_type == "CbpNet":
        logger.info("Using AdamGnT")
        optimizer = AdamGnT(trainable_params, lr=args.learning_rate, eps=1e-5)
        GnT = GnT(net=agent.actor.net, opt=optimizer,
                    replacement_rate=1e-3, decay_rate=0.99, device=device,
                    maturity_threshold=1000, util_type="contribution")
    if ("CKA" in args.method_type) and args.task_id > 1:
        logger.info(f"Create Alpha Optimizer for alpha training - learning rate: {args.alpha_learning_rate}")
        ao_exist = True
        alpha_params = [param for name, param in agent.named_parameters() if param.requires_grad and "alpha" in name]
        alpha_optimizer = optim.Adam(alpha_params, lr=args.alpha_learning_rate, eps=1e-5)
    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    loop = tqdm(range(1, args.num_iterations + 1))
    for iteration in loop:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            if ao_exist:
                alpha_lrnow = frac * args.alpha_learning_rate
                alpha_optimizer.param_groups[0]["lr"] = alpha_lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if (
                    args.track
                    and args.method_type == "CompoNet"
                    and global_step % 100 == 0
                ):
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0,
                        log_writter=writer,
                        global_step=global_step,
                        prevs_to_noise=args.prevs_to_noise,
                    )
                elif args.method_type == "CompoNet":
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0, prevs_to_noise=args.prevs_to_noise
                    )
                elif "CKA-RL" in args.method_type:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0, 
                        log_writter=writer, 
                        global_step=global_step
                    )
                else:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs / 255.0
                    )

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logger.debug(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        logs["global_step"].append(global_step)
                        logs["episodic_return"].append(info["episode"]["r"].item())
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs / 255.0).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if args.method_type == "CompoNet":
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds] / 255.0,
                        b_actions.long()[mb_inds],
                        prevs_to_noise=args.prevs_to_noise,
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds] / 255.0, b_actions.long()[mb_inds]
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                if ao_exist:
                    alpha_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if args.method_type == "PackNet":
                    if global_step >= packnet_retrain_start:
                        agent.start_retraining()  # can be called multiple times, only the first counts
                    agent.before_update()
                optimizer.step()
                if ao_exist:
                    alpha_optimizer.step()
                # Continual Backpropagation: Selective Intialization
                if args.method_type == "CbpNet":
                    logger.debug("CbpNet: Selective Initialization")
                    GnT.gen_and_test(agent.actor.get_activations())
                
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        if ao_exist:
            writer.add_scalar(
                "charts/alpha_learning_rate", alpha_optimizer.param_groups[0]["lr"], global_step
            )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # logger.debug(f"SPS:{int(global_step / (time.time() - start_time))}")
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        
        # show SLS + return
        loop.set_postfix(SPS=int(global_step / (time.time() - start_time)),R=logs["episodic_return"][-1])
        
    envs.close()
    writer.close()
    
    if ao_exist:
        logger.info(f"final alpha : {agent.alpha.data}")
        
    import pandas as pd
    df = pd.DataFrame(logs)
    if args.tag is not None:
        log_dir = f"./data/{env_name}/{args.tag}/{args.method_type}/{args.task_id}"  
        os.makedirs(log_dir, exist_ok=True)
        df.to_csv(f"{log_dir}/returns.csv", index=False)
    
        agent.save(dirname=f"./agents/{env_name}/{args.tag}/{run_name}") # ./agents/Freeway/tag/run_name
