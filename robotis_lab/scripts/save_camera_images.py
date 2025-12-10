# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to capture and save RGB and Depth images from the OpenManipulator-X camera."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.cm as cm
from PIL import Image

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Capture images from OpenManipulator-X camera.")
parser.add_argument("--task", type=str, default="RobotisLab-Lift-Cube-OpenManipulatorX-Play-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--output_dir", type=str, default="captured_images", help="Directory to save images.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import robotis_lab  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg=None):
    """Capture images."""
    
    # Override num_envs
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run a few steps to ensure camera data is populated
    print("[INFO] Stepping simulation to warm up camera...")
    # Create zero actions
    actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_space.shape[1], device=env.unwrapped.device)
    
    for _ in range(20):
        env.step(actions)
    
    # Access the camera
    try:
        # Access the scene from the unwrapped environment
        # Note: env might be wrapped, so we use env.unwrapped
        camera = env.unwrapped.scene["static_camera"]
    except KeyError:
        print(f"[ERROR] Could not find 'static_camera' in the scene. Available sensors: {list(env.unwrapped.scene.sensors.keys())}")
        env.close()
        return

    # Get data
    print("[INFO] Capturing images...")
    
    rgb_data = camera.data.output["rgb"]
    depth_data = camera.data.output["distance_to_image_plane"]
    
    # Data shape: (num_envs, height, width, channels)
    env_id = 0
    
    # RGB
    rgb_img_tensor = rgb_data[env_id] # (H, W, 4) usually RGBA
    rgb_np = rgb_img_tensor.cpu().numpy()
    
    # Remove alpha if present
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]
        
    # Depth
    depth_img_tensor = depth_data[env_id] # (H, W, 1)
    depth_np = depth_img_tensor.squeeze().cpu().numpy()
    
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    # Save RGB
    rgb_image = Image.fromarray(rgb_np.astype(np.uint8))
    rgb_path = os.path.join(args_cli.output_dir, "rgb_image.png")
    rgb_image.save(rgb_path)
    print(f"[INFO] Saved RGB image to {rgb_path}")
    
    # Save Depth (Visualization)
    # Handle infs (sky/far plane) - replace with max valid depth or 0
    valid_mask = np.isfinite(depth_np)
    if valid_mask.any():
        max_val = depth_np[valid_mask].max()
        min_val = depth_np[valid_mask].min()
        
        # Replace inf with max_val
        depth_vis = np.copy(depth_np)
        depth_vis[~valid_mask] = max_val
        
        # Normalize 0-1
        if max_val > min_val:
            depth_norm = (depth_vis - min_val) / (max_val - min_val)
        else:
            depth_norm = np.zeros_like(depth_vis)
            
        # Apply rainbow colormap (jet is common for depth)
        # cm.jet returns RGBA float 0-1
        depth_colored = cm.jet(depth_norm)
        # Convert to RGB uint8 0-255
        depth_vis = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    else:
        depth_vis = np.zeros((*depth_np.shape, 3), dtype=np.uint8)

    depth_image = Image.fromarray(depth_vis)
    depth_vis_path = os.path.join(args_cli.output_dir, "depth_vis.png")
    depth_image.save(depth_vis_path)
    print(f"[INFO] Saved depth visualization to {depth_vis_path}")
    
    # Save Raw Depth
    depth_raw_path = os.path.join(args_cli.output_dir, "depth_raw.npy")
    np.save(depth_raw_path, depth_np)
    print(f"[INFO] Saved raw depth data to {depth_raw_path}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
