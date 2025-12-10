# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
import pickle

def dump_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_lab  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    



    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # [FIX] Detect PixelActionWrapper to get the correct Discrete action space
    # SkrlVecEnvWrapper might report the base env's Box space, so we need to find the wrapper manually
    actual_action_space = env.action_space
    if isinstance(env.action_space, gym.spaces.Box):
        # Try to find PixelActionWrapper in the chain
        check_env = env
        pixel_wrapper_found = False
        while hasattr(check_env, "env"):
            if hasattr(check_env, "num_actions"): # PixelActionWrapper has this attribute
                print(f"[INFO] Found PixelActionWrapper with {check_env.num_actions} actions. Using Discrete action space for Agent.")
                actual_action_space = gym.spaces.Discrete(check_env.num_actions)
                pixel_wrapper_found = True
                break
            check_env = check_env.env
            
        if not pixel_wrapper_found:
             print("[WARNING] Could not find PixelActionWrapper. Agent might receive Box action space.")

    print(f"[DEBUG] Env Action Space: {env.action_space}")
    print(f"[DEBUG] Agent Action Space: {actual_action_space}")

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    
    # Check if we are using the custom RGBD_DQN
    if "models" in agent_cfg and agent_cfg["models"]["policy"]["class"] == "RGBD_DQN":
        from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift_visual_dqn.agents.skrl_dqn_model import RGBD_DQN
        from skrl.agents.torch.dqn import DQN
        from skrl.trainers.torch import SequentialTrainer
        from skrl.memories.torch import RandomMemory
        
        # Update config with actual class for Agent instantiation
        agent_cfg["models"]["policy"]["class"] = RGBD_DQN
        agent_cfg["models"]["target"]["class"] = RGBD_DQN
        if "agent" in agent_cfg and "models" in agent_cfg["agent"]:
             agent_cfg["agent"]["models"]["policy"]["class"] = RGBD_DQN
             agent_cfg["agent"]["models"]["target"]["class"] = RGBD_DQN
             
        # Initialize models
        models = {}
        models["q_network"] = RGBD_DQN(env.observation_space, actual_action_space, env.device)
        models["target_q_network"] = RGBD_DQN(env.observation_space, actual_action_space, env.device)
        
        # Initialize memory
        # Config: {"class": "RandomMemory", "memory_size": 10000}
        memory_size = agent_cfg["agent"]["memory"]["memory_size"]
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
        
        # Initialize agent
        agent = DQN(models=models,
                    memory=memory,
                    cfg=agent_cfg["agent"],
                    observation_space=env.observation_space,
                    action_space=actual_action_space,
                    device=env.device)
        
        # Initialize Trainer manually to avoid Runner config parsing issues
        trainer = SequentialTrainer(cfg=agent_cfg["trainer"], env=env, agents=agent)
        
        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            agent.load(resume_path)
            
        # run training
        trainer.train()
        
    else:
        runner = Runner(env, agent_cfg)

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # run training
        runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        raise e
    finally:
        # close sim app
        print("[INFO] Closing simulation app...")
        simulation_app.close()
