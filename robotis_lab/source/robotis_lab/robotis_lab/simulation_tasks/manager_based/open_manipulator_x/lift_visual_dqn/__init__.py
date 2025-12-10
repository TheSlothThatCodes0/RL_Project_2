# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import lift_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotisLab-Lift-OpenManipulatorX-Visual-DQN-v0",
    entry_point=f"{__name__}.env_wrapper:make_wrapped_env",
    kwargs={
        "env_cfg_entry_point": lift_env_cfg.OpenManipulatorXVisualLiftDQNEnvCfg,
        "skrl_cfg_entry_point": f"{__name__}.agents.skrl_dqn_cfg:LiftDQNAgentCfg",
        "skrl_dqn_cfg_entry_point": f"{__name__}.agents.skrl_dqn_cfg:LiftDQNAgentCfg",
    },
    disable_env_checker=True,
)
