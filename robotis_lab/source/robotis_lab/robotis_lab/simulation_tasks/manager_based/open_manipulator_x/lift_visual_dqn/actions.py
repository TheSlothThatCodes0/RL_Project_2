# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

@configclass
class DiscreteJointActionCfg:
    """Configuration for discrete joint action term."""
    class_type = None # Will be set to DiscreteJointAction
    asset_name: str = "robot"
    joint_names: list[str] | str = ".*"
    scale: float = 0.1

class DiscreteJointAction(ActionTerm):
    """Discrete action term that maps an integer index to joint position deltas."""

    cfg: DiscreteJointActionCfg

    def __init__(self, cfg: DiscreteJointActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Action mapping:
        # 0: No-op
        # 1..N: +Scale for Joint i
        # N+1..2N: -Scale for Joint i
        # Total actions: 2*N + 1
        self.num_actions = 2 * self._num_joints + 1

    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        # actions is (num_envs, 1)
        actions = actions.squeeze(-1).long()
        
        # Current joint positions
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]
        target_pos = current_pos.clone()
        
        # Apply deltas
        # 1..N -> Positive
        pos_mask = (actions >= 1) & (actions <= self._num_joints)
        if pos_mask.any():
            joint_idx = actions[pos_mask] - 1
            # We need to apply to specific joints per env
            # This is tricky vectorized.
            # Create a delta tensor
            delta = torch.zeros_like(target_pos)
            # Scatter add?
            # delta[env_idx, joint_idx] = scale
            # But we have a mask.
            
            # Simpler:
            # Create one-hot like mask
            # actions (num_envs,)
            # We want (num_envs, num_joints)
            
            # Map action to joint index and sign
            # 0 -> None
            # 1 -> J0 +
            # 2 -> J1 +
            # ...
            # N+1 -> J0 -
            # ...
            
        # Let's do it fully vectorized
        deltas = torch.zeros_like(target_pos)
        
        # Positive moves
        # actions 1..N maps to joint 0..N-1
        pos_indices = (actions - 1)
        valid_pos = (actions >= 1) & (actions <= self._num_joints)
        
        if valid_pos.any():
            # Create a mask for the joints to move
            # We can use scatter
            src = torch.full((valid_pos.sum(),), self.cfg.scale, device=self.device)
            # We need to scatter into deltas[valid_pos]
            # deltas[valid_pos].scatter_(1, pos_indices[valid_pos].unsqueeze(1), src.unsqueeze(1))
            # Wait, scatter_ expects src to be same size?
            
            # Let's use one_hot
            one_hot = torch.nn.functional.one_hot(pos_indices[valid_pos], num_classes=self._num_joints).float()
            deltas[valid_pos] += one_hot * self.cfg.scale

        # Negative moves
        # actions N+1..2N maps to joint 0..N-1
        neg_indices = (actions - (self._num_joints + 1))
        valid_neg = (actions >= self._num_joints + 1) & (actions <= 2 * self._num_joints)
        
        if valid_neg.any():
            one_hot = torch.nn.functional.one_hot(neg_indices[valid_neg], num_classes=self._num_joints).float()
            deltas[valid_neg] -= one_hot * self.cfg.scale
            
        target_pos += deltas
        
        # Clip to limits
        # We need limits. self._asset.data.soft_joint_pos_limits
        # ...
        
        return target_pos

    def apply_actions(self) -> None:
        # Set joint position targets
        self._asset.set_joint_position_target(self._raw_actions[:, self._joint_ids], joint_ids=self._joint_ids)

DiscreteJointActionCfg.class_type = DiscreteJointAction
