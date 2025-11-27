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

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

class LiftPPOAgentCfg:
    """Configuration for the PPO agent."""
    
    def __init__(self):
        self.config = PPO_DEFAULT_CONFIG.copy()
        self.config["rollouts"] = 24
        self.config["learning_epochs"] = 5 # Aligned with RSL-RL
        self.config["mini_batches"] = 4
        self.config["discount_factor"] = 0.98 # Aligned with RSL-RL
        self.config["lambda"] = 0.95
        self.config["learning_rate"] = 1.0e-4 # Aligned with RSL-RL
        self.config["learning_rate_scheduler"] = "KLAdaptiveLR"
        self.config["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01} # Aligned with RSL-RL desired_kl
        self.config["grad_norm_clip"] = 1.0
        self.config["ratio_clip"] = 0.2
        self.config["value_loss_scale"] = 1.0
        self.config["entropy_loss_scale"] = 0.006 # Aligned with RSL-RL
        self.config["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.1 # Scale rewards
