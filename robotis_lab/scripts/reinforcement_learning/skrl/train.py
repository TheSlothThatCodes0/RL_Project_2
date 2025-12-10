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
import numpy as np

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
import signal
import glob
import sys
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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

class Logger(object):
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import robotis_lab  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

def save_checkpoint_and_plot(agent, log_dir):
    print("\n[INFO] Saving agent checkpoint...")
    try:
        # Flush TensorBoard writer if it exists to ensure logs are on disk
        if hasattr(agent, "writer") and agent.writer is not None:
            print("[INFO] Flushing TensorBoard writer...")
            agent.writer.flush()
            
        checkpoint_path = os.path.join(log_dir, "checkpoints", "manual_save_agent.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        agent.save(checkpoint_path)
        print(f"[INFO] Agent saved to {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save checkpoint: {e}")
    
    # Plotting
    try:
        print("[INFO] Generating reward graph...")
        
        # Find tfevents file
        event_files = glob.glob(os.path.join(log_dir, "**", "*tfevents*"), recursive=True)
        if not event_files:
            print("[WARNING] No TensorBoard event file found. Cannot plot graph.")
            return

        # Use the most recent one
        event_file = max(event_files, key=os.path.getmtime)
        print(f"[INFO] Reading logs from: {event_file}")
        
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        
        # Check for available reward tags
        reward_tag = None
        possible_tags = ["Reward / Total reward (mean)", "Reward / Mean reward", "Reward / Instantaneous reward (mean)"]
        
        for tag in possible_tags:
            if tag in tags:
                reward_tag = tag
                break
                
        if reward_tag:
            print(f"[INFO] Plotting using tag: {reward_tag}")
            events = ea.Scalars(reward_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            if len(steps) > 0:
                plt.figure(figsize=(10, 6))
                
                # Plot raw data
                plt.plot(steps, values, label="Mean Reward", alpha=0.3)
                
                # Calculate and plot moving average (window=100)
                window_size = 100
                if len(values) >= window_size:
                    moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    ma_steps = steps[window_size-1:]
                    plt.plot(ma_steps, moving_avg, label=f"Moving Avg ({window_size})", color='red', linewidth=2)
                
                plt.xlabel("Timesteps")
                plt.ylabel("Reward")
                plt.title("Training Progress")
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(log_dir, "reward_graph.png")
                plt.savefig(plot_path)
                print(f"[INFO] Reward graph saved to {plot_path}")
                plt.close()
            else:
                print("[WARNING] Found 'Reward / Mean reward' tag but no data points.")
        else:
            print(f"[WARNING] 'Reward / Mean reward' not found in logs. Available tags: {tags}")
            print("[INFO] Note: For very short runs, TensorBoard might not have written data yet.")
            
    except Exception as e:
        print(f"[ERROR] Failed to generate graph: {e}")


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
    # Set checkpoint interval to 200 steps as requested
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 200
    
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
    
    # Create log directory and redirect stdout/stderr
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "output.log")
    sys.stdout = Logger(sys.stdout, log_file_path)
    sys.stderr = Logger(sys.stderr, log_file_path)
    print(f"[INFO] Saving console logs to: {log_file_path}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "env.pkl"), 'wb') as f:
        pickle.dump(env_cfg, f)
    with open(os.path.join(log_dir, "params", "agent.pkl"), 'wb') as f:
        pickle.dump(agent_cfg, f)

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

    print(f"[DEBUG] Env Observation Space BEFORE Skrl Wrapper: {env.observation_space}")

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    print(f"[DEBUG] Env Observation Space AFTER Skrl Wrapper: {env.observation_space}")

    # [FIX] Manually correct the Skrl Wrapper's observation space if it reverted to the original
    # This handles the case where SkrlVecEnvWrapper looks at unwrapped env and ignores PixelActionWrapper's changes
    if hasattr(env, "observation_space") and isinstance(env.observation_space, gym.spaces.Dict):
        # Check if "policy" key exists (standard Isaac Lab) or if it's flattened (Skrl Wrapper)
        target_space = env.observation_space
        if "policy" in env.observation_space.spaces:
            target_space = env.observation_space["policy"]
            
        if isinstance(target_space, gym.spaces.Dict):
            for key in ["rgb", "depth"]:
                if key in target_space.spaces:
                    space = target_space[key]
                    # Check if shape matches 224x224 (original) instead of 34x69 (cropped)
                    # Handle both (H, W, C) and (N, H, W, C)
                    h_idx = -3 if len(space.shape) >= 3 else 0
                    if len(space.shape) >= 3 and space.shape[h_idx] == 224:
                        print(f"[WARNING] Skrl Wrapper has incorrect shape for {key}: {space.shape}. Fixing it...")
                        # We know the correct shape from the PixelActionWrapper logic (hardcoded for now or inferred)
                        # ROI: 34x69
                        # Preserve N if it exists
                        if len(space.shape) == 4:
                            new_shape = (space.shape[0], 69, 34, space.shape[3])
                        else:
                            new_shape = (69, 34, space.shape[2])
                        
                        # Create new Box
                        from gymnasium.spaces import Box
                        new_box = Box(low=space.low.min(), high=space.high.max(), shape=new_shape, dtype=space.dtype)
                        
                        # Update the space in place
                        target_space.spaces[key] = new_box
                        print(f"[INFO] Fixed {key} space to: {new_box.shape}")

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
        # [FIX] Ensure memory replacement is False to avoid issues with dict observations if needed?
        # Actually, the issue is likely that RandomMemory doesn't support Dict observations well by default
        # or it flattens them.
        # However, skrl's RandomMemory should handle whatever space is passed to it.
        # But we need to make sure we pass the correct observation space to it if it needs it.
        # RandomMemory(memory_size, num_envs, device, export=False, replacement=True)
        # It doesn't take observation_space in init.
        memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
        
        # [CRITICAL FIX] Skrl's memory buffer might flatten the observation if not handled correctly.
        # But more importantly, when sampling from memory, it returns what was stored.
        # If the environment returns a dict, it should store a dict (or separate tensors).
        # The error happens in `act` which calls `compute`.
        # `act` is called during training (exploration) and evaluation.
        # If it fails after 1000 steps, it's likely when `learning_starts` triggers and it samples from memory?
        # Wait, the traceback says `single_agent_train` -> `self.agents.act`.
        # This is the exploration step, taking the current state from the environment.
        # If `states` passed to `act` is not a dict, then `env.reset()` or `env.step()` returned something else.
        # SkrlVecEnvWrapper might be flattening the observation!
        
        # Let's check SkrlVecEnvWrapper again. It calls `wrap_env`.
        # If `wrap_env` uses `IsaacLabWrapper`, it might be flattening.
        # We need to ensure the wrapper preserves the dictionary structure.
        # In `skrl.envs.wrappers.torch.isaaclab_envs.py`, `_tensorize_space` handles Dicts recursively.
        # But `step` and `reset` return the observation.
        
        # If the error happens after 1000 steps, it might be related to the first update?
        # "learning_starts" = 1000.
        # If it fails at step 1000, it's likely the training loop calling `agent.act` or `agent.record_transition`.
        # The traceback shows `self.agents.act(states, ...)` failing.
        # This `states` comes from `self.state` in the trainer.
        # `self.state` is updated by `self.env.reset()` or `self.env.step()`.
        
        # If `states` is not a dict, then the environment wrapper is returning a flattened tensor.
        # This happens if the observation space is NOT a Dict space according to the wrapper.
        # We patched the observation space in `train.py` AFTER creating the wrapper.
        # But maybe the wrapper cached the space or structure?
        
        # Let's verify if we need to pass `replacement=False` or similar to memory? No.
        
        # The issue is likely that `SkrlVecEnvWrapper` (which is `IsaacLabWrapper`)
        # might be converting the Dict observation into a flat tensor if it thinks it should.
        # But we saw `[DEBUG] Env Observation Space AFTER Skrl Wrapper: Dict(...)` in the logs.
        # So the wrapper *thinks* it's a Dict.
        
        # Wait, if `learning_starts` is 1000, then at step 1000, the agent starts training.
        # But `act` is called at every step.
        # Why did it work for 999 steps?
        # Ah, `act` is called to get the next action.
        # Maybe at step 1000, something changes?
        # The traceback shows `self.agents.act` calling `self.q_network.act`.
        # `states` is passed.
        
        # If `states` became a Tensor, it means `env.step()` returned a Tensor.
        # Why would `env.step()` return a Tensor suddenly?
        # Maybe `reset()` returns a Dict, but `step()` returns a Tensor?
        # Or maybe the `PixelActionWrapper` has a bug?
        
        # Let's look at `PixelActionWrapper.step`.
        # It returns `obs`. `obs` comes from `self.env.step()`.
        # `self.env` is the Isaac Lab env. It returns a Dict.
        # `PixelActionWrapper._crop_observation` modifies the Dict.
        # So it should return a Dict.
        
        # Is it possible that `skrl`'s `SequentialTrainer` modifies `self.state`?
        # In `single_agent_train`:
        # states, infos = self.env.reset()
        # for timestep in range(...):
        #     actions = self.agents.act(states, ...)
        #     next_states, rewards, terminated, truncated, infos = self.env.step(actions)
        #     ...
        #     states = next_states
        
        # If `next_states` is a Tensor, then `states` becomes a Tensor for the next iteration.
        # So `env.step()` must be returning a Tensor.
        
        # Why would `env.step()` return a Tensor?
        # The `SkrlVecEnvWrapper` (IsaacLabWrapper) `step` method:
        # return self._tensorize_observation(obs), ...
        # `_tensorize_observation` uses `self.observation_space` to decide how to format.
        # If `self.observation_space` is a Dict, it returns a Dict.
        
        # We patched `env.observation_space` in `train.py`.
        # BUT, `IsaacLabWrapper` might have internal logic or cached properties?
        # Actually, `IsaacLabWrapper` inherits from `Wrapper`.
        # It uses `self.observation_space`.
        
        # HYPOTHESIS: The `PixelActionWrapper` might be returning a flattened observation in `step` 
        # under some condition? No, it just calls `_crop_observation`.
        
        # HYPOTHESIS 2: The `SkrlVecEnvWrapper` is not seeing the patched observation space?
        # We patched `env.observation_space` which IS the wrapper instance.
        # So it should see it.
        
        # HYPOTHESIS 3: The `states` variable in the trainer is being corrupted?
        # Or maybe `agent.act` modifies it in place? Unlikely.
        
        # Let's add a check in the loop? We can't easily modify the trainer code.
        # But we can wrap the environment again to ensure it always returns a Dict?
        
        # Or maybe the issue is that `skrl`'s `DQN` agent expects something else?
        # No, we wrote the model to expect a Dict.
        
        # Let's try to force the environment wrapper to respect the Dict space.
        # We can subclass `SkrlVecEnvWrapper` or wrap it.
        
        # Better yet, let's add a debug print in `PixelActionWrapper` to see what it returns.
        # And maybe we can add a "SafetyWrapper" that ensures output is a Dict.
        
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
            
        # Define signal handler for robust saving
        def handle_sigint(sig, frame):
            print("\n[INFO] SIGINT (Ctrl+C) detected via signal handler.")
            save_checkpoint_and_plot(agent, log_dir)
            print("[INFO] Exiting process immediately.")
            os._exit(0)
            
        signal.signal(signal.SIGINT, handle_sigint)
            
        # run training
        trainer.train()
        
    else:
        runner = Runner(env, agent_cfg)

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # Define signal handler for robust saving
        def handle_sigint(sig, frame):
            print("\n[INFO] SIGINT (Ctrl+C) detected via signal handler.")
            save_checkpoint_and_plot(runner.agent, log_dir)
            print("[INFO] Exiting process immediately.")
            os._exit(0)
            
        signal.signal(signal.SIGINT, handle_sigint)

        # run training
        runner.run()
        
        # Save and plot at the end of training
        save_checkpoint_and_plot(runner.agent, log_dir)

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
