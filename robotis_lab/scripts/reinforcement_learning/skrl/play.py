# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import robotis_lab  # noqa: F401
# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # [FIX] Detect PixelActionWrapper to get the correct Discrete action space
    actual_action_space = env.action_space
    if isinstance(env.action_space, gym.spaces.Box):
        # Try to find PixelActionWrapper in the chain
        check_env = env
        pixel_wrapper_found = False
        while hasattr(check_env, "env"):
            if hasattr(check_env, "num_actions"): # PixelActionWrapper has this attribute
                print(f"[INFO] Found PixelActionWrapper with {check_env.num_actions} actions. Using Discrete action space for Agent.")
                actual_action_space = gym.spaces.Discrete(check_env.num_actions)
                
                # CRITICAL FIX: We must also update the wrapper's action space so it doesn't try to reshape
                # the single integer action into the original Box shape (8,)
                try:
                    env.action_space = actual_action_space
                except AttributeError:
                    # If action_space is a property without a setter, try setting the private attribute
                    # This is common in some wrappers that expose action_space as a property
                    if hasattr(env, "_action_space"):
                        env._action_space = actual_action_space
                    
                    # Also try to set it on the inner wrapper we found
                    try:
                        check_env.action_space = actual_action_space
                    except:
                        pass
                
                pixel_wrapper_found = True
                break
            check_env = check_env.env
            
        if not pixel_wrapper_found:
             print("[WARNING] Could not find PixelActionWrapper. Agent might receive Box action space.")

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    
    # Check if we are using the custom RGBD_DQN
    if "models" in experiment_cfg and experiment_cfg["models"]["policy"]["class"] == "RGBD_DQN":
        from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift_visual_dqn.agents.skrl_dqn_model import RGBD_DQN
        from skrl.agents.torch.dqn import DQN
        from skrl.trainers.torch import SequentialTrainer
        from skrl.memories.torch import RandomMemory
        
        # Update config with actual class for Agent instantiation
        experiment_cfg["models"]["policy"]["class"] = RGBD_DQN
        experiment_cfg["models"]["target"]["class"] = RGBD_DQN
        if "agent" in experiment_cfg and "models" in experiment_cfg["agent"]:
             experiment_cfg["agent"]["models"]["policy"]["class"] = RGBD_DQN
             experiment_cfg["agent"]["models"]["target"]["class"] = RGBD_DQN
             
        # Initialize models
        models = {}
        models["q_network"] = RGBD_DQN(env.observation_space, actual_action_space, env.device)
        models["target_q_network"] = RGBD_DQN(env.observation_space, actual_action_space, env.device)
        
        # Initialize memory (dummy for eval)
        memory = RandomMemory(memory_size=10, num_envs=env.num_envs, device=env.device)
        
        # Initialize agent
        agent = DQN(models=models,
                    memory=memory,
                    cfg=experiment_cfg["agent"],
                    observation_space=env.observation_space,
                    action_space=actual_action_space,
                    device=env.device)
        
        # Initialize Trainer manually (used as runner wrapper here)
        # We can't use SequentialTrainer directly for eval loop below easily without refactoring
        # But we can just use the agent directly.
        
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
        agent.set_running_mode("eval")
        
        # Create a dummy runner object to satisfy the loop below
        class DummyRunner:
            def __init__(self, agent):
                self.agent = agent
        runner = DummyRunner(agent)
        
    else:
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                # DQN act() returns (actions, None, None)
                # PPO act() returns (actions, log_prob, outputs) where outputs is a dict
                if isinstance(outputs, tuple):
                    # Check if the last element is None (typical for DQN)
                    if outputs[-1] is None:
                        actions = outputs[0]
                    else:
                        # PPO case
                        actions = outputs[-1].get("mean_actions", outputs[0])
                else:
                    # Fallback
                    actions = outputs
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
