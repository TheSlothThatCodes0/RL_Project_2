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

import torch
import torch.nn as nn
import gymnasium as gym
from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

# Define the Unified Vision + PPO Policy Architecture
class VisualPPOPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # 1. Visual Encoder (CNN) - Modified ResNet-18 style or NatureCNN
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
        # Note: Isaac Lab might flatten this, so we might need to reshape
        # Assuming inputs["states"] contains the flattened dict or we handle it in wrapper
        
        # For this blueprint, we assume the wrapper has structured the input as:
        # inputs["visual"] -> (B, 4, 84, 84)
        # inputs["proprio"] -> (B, 8)
        
        visual = inputs.get("visual")
        proprio = inputs.get("proprio")

        # CNN Pass
        visual_feat = self.cnn(visual)
        
        # Fusion
        fused = torch.cat([visual_feat, proprio], dim=1)
        
        # MLP Pass
        x = self.net(fused)
        
        # Output
        return self.mean_layer(x), self.log_std_parameter, {}

class VisualPPOValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # Same architecture as Policy but outputs scalar value
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
        visual_feat = self.cnn(visual)
        fused = torch.cat([visual_feat, proprio], dim=1)
        return self.net(fused), {}

# Training Script
if __name__ == "__main__":
    import argparse
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.envs.loaders.torch import load_isaaclab_env
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="RobotisLab-Lift-OpenManipulatorX-Visual-PPO-v0")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    # Load Environment
    env = load_isaaclab_env(task_name=args.task, num_envs=args.num_envs, headless=args.headless)
    
    # WRAPPER: Handle Observation Reshaping for CNN
    # This is a simplified inline wrapper logic. In production, use a proper Gym Wrapper class.
    # We need to ensure 'env' returns a dict with 'visual' and 'proprio' keys that match the model expectation.
    # Isaac Lab returns a flat tensor for 'policy' group if concatenate_terms=True.
    # Since we set concatenate_terms=False in config, we get a dict.
    
    # Configure PPO
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 24
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = "KLAdaptiveLR"
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_loss_scale"] = 1.0
    cfg["entropy_loss_scale"] = 0.0
    
    # Instantiate Models
    models = {}
    models["policy"] = VisualPPOPolicy(env.observation_space, env.action_space, env.device)
    models["value"] = VisualPPOValue(env.observation_space, env.action_space, env.device)

    # Instantiate Agent
    agent = PPO(models=models,
                memory=None, 
                cfg=cfg, 
                observation_space=env.observation_space, 
                action_space=env.action_space, 
                device=env.device)

    # Trainer
    trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)
    trainer.train()
