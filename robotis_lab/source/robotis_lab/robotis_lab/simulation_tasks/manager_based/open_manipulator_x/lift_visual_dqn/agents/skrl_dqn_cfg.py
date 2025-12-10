from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
from .skrl_dqn_model import RGBD_DQN
from omegaconf import OmegaConf, DictConfig

# Convert OmegaConf to dict if necessary
if isinstance(DQN_DEFAULT_CONFIG, DictConfig):
    _dqn_config = OmegaConf.to_container(DQN_DEFAULT_CONFIG, resolve=True)
elif hasattr(DQN_DEFAULT_CONFIG, "copy"):
    _dqn_config = DQN_DEFAULT_CONFIG.copy()
else:
    _dqn_config = dict(DQN_DEFAULT_CONFIG)

LiftDQNAgentCfg = {
    "agent": _dqn_config,
    "seed": 42,
}
# Ensure models are at the top level for Runner if needed, but usually they are under agent
# However, skrl Runner expects "models" key in the root cfg if it's not found in agent?
# Wait, the error says "No 'models' are defined in cfg".
# The Runner looks for cfg["models"] OR cfg["agent"]["models"] depending on how it's called.
# But here we are passing agent_cfg which IS LiftDQNAgentCfg.

# Let's check skrl Runner code logic.
# It seems it might be looking for "models" at the top level if we are using a specific runner type?
# Or maybe the structure is slightly off.

# Let's add "models" to the top level as well just in case, pointing to the same dict.
LiftDQNAgentCfg["models"] = {
    "policy": {
        "class": "RGBD_DQN",
        "clip_actions": False,
    },
    "target": {
        "class": "RGBD_DQN",
        "clip_actions": False,
    },
}

LiftDQNAgentCfg["agent"]["class"] = "DQN"
LiftDQNAgentCfg["agent"]["models"] = LiftDQNAgentCfg["models"]

# Memory
# Note: Storing 100k 224x224 RGBD images requires ~35GB RAM. Reduced to 1000 (~350MB per env).
# With 16 envs, this is ~5.6GB.
LiftDQNAgentCfg["agent"]["memory"] = {"class": "RandomMemory", "memory_size": 1000}

# Training
LiftDQNAgentCfg["agent"]["batch_size"] = 32 # Reduced batch size for heavier model
LiftDQNAgentCfg["agent"]["random_timesteps"] = 1000
LiftDQNAgentCfg["agent"]["learning_starts"] = 1000
LiftDQNAgentCfg["agent"]["update_interval"] = 1
LiftDQNAgentCfg["agent"]["target_update_interval"] = 1000

# Exploration
LiftDQNAgentCfg["agent"]["exploration"] = {
    "initial_epsilon": 1.0,
    "final_epsilon": 0.05,
    "timesteps": 50000,
}

# Trainer
LiftDQNAgentCfg["trainer"] = {
    "timesteps": 100000,
    "headless": True,
}
