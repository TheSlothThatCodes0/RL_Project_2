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

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
import torch

def action_rate_l2(env: ManagerBasedRLEnv, command_name: str = "arm_action") -> torch.Tensor:
    """Penalize the rate of change of the actions to encourage smooth motion."""
    # The actions are stored in the environment's action manager
    # We need to access the specific action term
    # This is a simplified implementation assuming the action term stores its history or we use the env's last action
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above a minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Target is the object position
    target_pos = object.data.root_pos_w
    # EE position
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    
    distance = torch.norm(target_pos - ee_pos, dim=-1)
    return 1.0 / (1.0 + (distance / std).pow(2))
