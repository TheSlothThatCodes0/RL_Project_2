# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv
from .pixel_wrapper import PixelActionWrapper

def make_wrapped_env(cfg, **kwargs):
    """Entry point to create the environment and wrap it."""
    env = ManagerBasedRLEnv(cfg=cfg, **kwargs)
    
    # Extract camera config
    # The cfg object passed here is an instance of OpenManipulatorXVisualLiftDQNEnvCfg
    # We need to access the camera config. 
    # In the config class, it's defined as self.scene.static_camera
    
    # However, cfg might be the *instance* of the config class.
    camera_cfg = cfg.scene.static_camera
    
    return PixelActionWrapper(env, camera_cfg)
