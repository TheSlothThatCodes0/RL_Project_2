import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp
from .lift_env_cfg import LiftEnvCfg
from robotis_lab.assets.robots.open_manipulator_x import OPEN_MANIPULATOR_X_CFG

@configclass
class OpenManipulatorXCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Robot
        self.scene.robot = OPEN_MANIPULATOR_X_CFG.replace(prim_path="{ENV_REGEX_NS}/open_manipulator_x")

        # Set actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-4]"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_left_joint", "gripper_right_joint"],
            open_command_expr={"gripper_left_joint": 0.019, "gripper_right_joint": 0.019},
            close_command_expr={"gripper_left_joint": -0.01, "gripper_right_joint": -0.01},
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "link5"
        self.commands.object_pose.ranges.pos_x = (0.15, 0.25)
        self.commands.object_pose.ranges.pos_y = (-0.15, 0.15)
        self.commands.object_pose.ranges.pos_z = (0.15, 0.25)

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            
            ),
        )

        # Adjust reset noise to keep object within reach
        self.events.reset_object_position.params["pose_range"]["x"] = (0.0, 0.0)
        self.events.reset_object_position.params["pose_range"]["y"] = (0.0, 0.0)

        # EE Frame
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/open_manipulator_x/link1", # Attach to base link (link1)
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/open_manipulator_x/link5",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.126, 0.0, 0.0],
                    ),
                ),
            ],
        )

@configclass
class OpenManipulatorXCubeLiftEnvCfg_PLAY(OpenManipulatorXCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
