from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Reward the agent for lifting the object above a minimal height.

    *Only if* it is within a certain distance from the end-effector.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Get object height (z position in world frame)
    obj_height = object.data.root_pos_w[:, 2]  # (num_envs,)

    # Get positions
    obj_pos = object.data.root_pos_w  # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Compute Euclidean distance between object and end-effector
    dist = torch.norm(obj_pos - ee_pos, dim=1)  # (num_envs,)

    # Reward is 1.0 if object is above minimal height AND within_reach to EE
    lifted = obj_height > minimal_height
    within_reach = dist < distance_threshold

    reward = torch.where(lifted & within_reach, 1.0, 0.0)

    return reward


def object_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    xy_diff_threshold: float = 0.03,
    z_diff_threshold: float = 0.03,
    gripper_close_threshold: float = 0.005,
    gripper_fully_closed_threshold: float = -0.005,
) -> torch.Tensor:
    """
    Reward function for detecting if the object is being grasped.

    Combines end-effector proximity (in XY and Z) and gripper closure conditions.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute the distance between end-effector and object
    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]

    # XY-plane distance
    xy_dist = torch.linalg.vector_norm(object_pos[:, :2] - end_effector_pos[:, :2], dim=1)
    # Z-axis distance
    z_dist = torch.abs(object_pos[:, 2] - end_effector_pos[:, 2])

    # Check if gripper joints are closed beyond threshold (i.e. not fully open)
    # AND not fully closed (i.e. holding something)
    # For OMX, closing means decreasing value (from 0.019 to -0.01)
    
    # Check if joints are less than "close threshold" (started closing)
    gripper_closing = torch.logical_and(
        robot.data.joint_pos[:, -1] <= gripper_close_threshold,
        robot.data.joint_pos[:, -2] <= gripper_close_threshold,
    )
    
    # Check if joints are greater than "fully closed threshold" (didn't close all the way)
    gripper_holding = torch.logical_and(
        robot.data.joint_pos[:, -1] >= gripper_fully_closed_threshold,
        robot.data.joint_pos[:, -2] >= gripper_fully_closed_threshold,
    )
    
    valid_gripper_state = torch.logical_and(gripper_closing, gripper_holding)

    # Combine all conditions
    is_grasped = torch.logical_and(xy_dist < xy_diff_threshold, z_dist < z_diff_threshold)
    is_grasped = torch.logical_and(is_grasped, valid_gripper_state)

    return is_grasped.float()


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,  # standard deviation
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)

    reward = torch.exp(-0.5 * (distance / std) ** 2)

    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
