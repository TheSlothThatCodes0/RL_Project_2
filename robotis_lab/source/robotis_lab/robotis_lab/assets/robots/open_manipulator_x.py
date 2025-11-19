import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

OPEN_MANIPULATOR_X_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/pi0/RL_GOOD/data/usd1_2500nf.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "gripper_left_joint": 0.0,
            "gripper_right_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            velocity_limit_sim=4.8,
            effort_limit_sim=100.0,
            stiffness=100,
            damping=15,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_left_joint", "gripper_right_joint"],
            velocity_limit_sim=4.8,
            effort_limit_sim=100.0,
            stiffness=20,
            damping=5,
        ),
    },
)
