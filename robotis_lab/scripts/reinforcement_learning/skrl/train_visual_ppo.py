# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to train Visual PPO agent with skrl for OpenManipulator-X.
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a Visual PPO agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="RobotisLab-Lift-OpenManipulatorX-Visual-PPO-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--cnn_type", type=str, default="nature", choices=["nature", "resnet"], help="Type of CNN architecture to use: 'nature' or 'resnet'.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video or if task is visual
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import torch.nn as nn
import torchvision.models as models
import gymnasium as gym
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.ppo import PPO
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

# Import the task to register it
import robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift_visual_ppo  # noqa: F401

# Define the Unified Vision + PPO Policy Architecture
class VisualPPOPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, cnn_type="nature", clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # 1. Visual Encoder (CNN)
        if cnn_type == "resnet":
            # ResNet-18
            # Input: (4, 84, 84) -> RGB (3) + Depth (1)
            # We use standard ResNet18 but modify the first layer to accept 4 channels
            # Note: We assume training from scratch, so no pre-trained weights
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)
                
            # Original conv1: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove the fully connected layer (fc) to keep features
            # The output of the rest is (B, 512, 1, 1) after AdaptiveAvgPool2d
            self.cnn = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        else:
            # NatureCNN
            # Input: (4, 84, 84) -> RGB (3) + Depth (1)
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        # Compute CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 84, 84)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # 2. Proprioception Encoder
        # Joint Pos (4) + Joint Vel (4) = 8
        self.proprio_dim = 8 
        
        # 3. Fusion MLP
        self.fusion_dim = cnn_out_size + self.proprio_dim
        self.net = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        # 4. PPO Actor Head (Mean)
        self.mean_layer = nn.Linear(128, self.num_actions)
        
        # 5. Log Std Parameter
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Unpack dictionary observation
        # The wrapper below ensures we get a dict with 'visual' and 'proprio'
        
        visual = inputs.get("visual")
        proprio = inputs.get("proprio")

        # Permute visual to (B, C, H, W) if needed
        # Isaac Lab cameras usually return (B, H, W, C) or (B, C, H, W) depending on config
        # We assume (B, 4, 84, 84) here. If not, we permute.
        if visual.shape[-1] == 4:
             visual = visual.permute(0, 3, 1, 2)

        # CNN Pass
        visual_feat = self.cnn(visual)
        
        # Fusion
        fused = torch.cat([visual_feat, proprio], dim=1)
        
        # MLP Pass
        x = self.net(fused)
        
        # Output
        return self.mean_layer(x), self.log_std_parameter, {}

class VisualPPOValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, cnn_type="nature", clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Same architecture as Policy but outputs scalar value
        if cnn_type == "resnet":
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)
            resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, 84, 84)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        self.proprio_dim = 8 
        self.fusion_dim = cnn_out_size + self.proprio_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1) # Value output
        )

    def compute(self, inputs, role):
        visual = inputs.get("visual")
        proprio = inputs.get("proprio")
        
        if visual.shape[-1] == 4:
             visual = visual.permute(0, 3, 1, 2)
             
        visual_feat = self.cnn(visual)
        fused = torch.cat([visual_feat, proprio], dim=1)
        return self.net(fused), {}

class VisualObservationWrapper(gym.Wrapper):
    """
    Custom wrapper to format Isaac Lab observations for the Visual PPO Policy.
    Combines RGB and Depth into a single 'visual' tensor and extracts 'proprio'.
    """
    def __init__(self, env):
        super().__init__(env)
        # We need to define the observation space structure for SKRL
        # This assumes the underlying env returns a dict with 'policy' group containing the terms
        
        # Note: We construct the spaces manually based on known config
        self.observation_space = gym.spaces.Dict({
            "visual": gym.spaces.Box(low=0, high=1, shape=(4, 84, 84), dtype=float),
            "proprio": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(8,), dtype=float)
        })
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def _process_obs(self, obs):
        # obs is a dict from Isaac Lab. 
        # If concatenate_terms=False in PolicyCfg, obs['policy'] is a dict.
        policy_obs = obs['policy']
        
        # Extract terms
        # Note: Keys depend on the ObsTerm names in config
        rgb = policy_obs['rgb']     # (B, H, W, 3) or (B, 3, H, W)
        depth = policy_obs['depth'] # (B, H, W, 1) or (B, 1, H, W)
        joint_pos = policy_obs['joint_pos']
        joint_vel = policy_obs['joint_vel']
        
        # Concatenate RGB and Depth
        # Ensure they are (B, H, W, C) for concatenation then permute or keep as is
        # Isaac Lab images are usually (B, H, W, C)
        visual = torch.cat([rgb, depth], dim=-1) # (B, H, W, 4)
        
        # Concatenate Proprio
        proprio = torch.cat([joint_pos, joint_vel], dim=-1)
        
        return {
            "visual": visual,
            "proprio": proprio
        }

def main():
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.headless, num_envs=args_cli.num_envs)
    
    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap for SKRL compatibility (device handling, etc.)
    # We use the standard wrapper first to handle basic Isaac Lab <-> Gym interface
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    # Apply our custom Visual Wrapper ON TOP to format observations
    env = VisualObservationWrapper(env)

    set_seed(args_cli.seed)

    # Configure PPO
    # Load from the config class we created
    from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift_visual_ppo.agents.skrl_ppo_cfg import LiftPPOAgentCfg
    agent_cfg = LiftPPOAgentCfg().config
    
    # Instantiate Models
    models = {}
    models["policy"] = VisualPPOPolicy(env.observation_space, env.action_space, env.device, cnn_type=args_cli.cnn_type)
    models["value"] = VisualPPOValue(env.observation_space, env.action_space, env.device, cnn_type=args_cli.cnn_type)

    # Instantiate Agent
    agent = PPO(models=models,
                memory=None, 
                cfg=agent_cfg, 
                observation_space=env.observation_space, 
                action_space=env.action_space, 
                device=env.device)

    # Trainer
    trainer = SequentialTrainer(cfg=agent_cfg, env=env, agents=agent)
    trainer.train()

if __name__ == "__main__":
    main()
