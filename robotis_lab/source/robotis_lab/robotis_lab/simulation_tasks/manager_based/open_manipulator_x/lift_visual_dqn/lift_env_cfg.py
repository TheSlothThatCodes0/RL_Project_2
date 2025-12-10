# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp
from isaaclab.controllers import DifferentialIKControllerCfg

from . import actions

from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift.joint_pos_env_cfg import OpenManipulatorXCubeLiftEnvCfg
from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift.lift_env_cfg import ActionsCfg, ObservationsCfg

def get_rgb_image(env):
    return env.scene["static_camera"].data.output["rgb"]

def get_depth_image(env):
    return env.scene["static_camera"].data.output["distance_to_image_plane"]

@configclass
class VisualObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        rgb = ObsTerm(func=get_rgb_image)
        depth = ObsTerm(func=get_depth_image)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class PixelActionsCfg(ActionsCfg):
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint[1-4]"],
        body_name="link5",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        ),
        scale=1.0,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_left_joint", "gripper_right_joint"],
        open_command_expr={"gripper_left_joint": 0.019, "gripper_right_joint": 0.019},
        close_command_expr={"gripper_left_joint": -0.01, "gripper_right_joint": -0.01},
    )

@configclass
class OpenManipulatorXVisualLiftDQNEnvCfg(OpenManipulatorXCubeLiftEnvCfg):
    observations: VisualObservationsCfg = VisualObservationsCfg()
    actions: PixelActionsCfg = PixelActionsCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
        # Restore PixelActionsCfg because parent __post_init__ overrides it with JointPosition
        self.actions = PixelActionsCfg()
        
        # Modify existing static_camera instead of creating a new one
        self.scene.static_camera.update_period = 0.1
        self.scene.static_camera.height = 224
        self.scene.static_camera.width = 224
        self.scene.static_camera.data_types = ["rgb", "distance_to_image_plane"]
        self.scene.static_camera.spawn.clipping_range = (0.01, 3.0)
        
        # Ensure offset is correct
        self.scene.static_camera.offset.pos = (1.4, 0.05, 0.4)
        self.scene.static_camera.offset.rot = (0.54, 0.41, 0.44, 0.57)
        self.scene.static_camera.offset.convention = "opengl"
