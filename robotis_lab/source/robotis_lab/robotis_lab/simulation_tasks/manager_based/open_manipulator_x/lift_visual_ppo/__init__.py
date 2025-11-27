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

import gymnasium as gym

from . import lift_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotisLab-Lift-OpenManipulatorX-Visual-PPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_env_cfg.OpenManipulatorXVisualLiftEnvCfg,
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:LiftPPORunnerCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents.skrl_ppo_cfg:LiftPPOAgentCfg",
    },
    disable_env_checker=True,
)
