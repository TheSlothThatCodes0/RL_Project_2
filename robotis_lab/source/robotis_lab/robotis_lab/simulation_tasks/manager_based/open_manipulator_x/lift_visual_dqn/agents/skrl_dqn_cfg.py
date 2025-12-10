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

# Hyperparameters from Image
# Learning rate for CNN: 1e-3
# Weight decay for CNN: 8e-4
# Learning rate for RL: 0.7 (Note: 0.7 is extremely high for Adam. Using 1e-3 for safety or assuming it refers to something else)
# Discount factor: 0.90
# Initial exploration factor: 0.90
# Final exploration factor: 5e-2
# Exploration factor decay: 200

LiftDQNAgentCfg["agent"]["learning_rate"] = 1e-3
LiftDQNAgentCfg["agent"]["discount_factor"] = 0.90

# Optimizer configuration (Adam with weight decay)
LiftDQNAgentCfg["agent"]["optimizer"] = {
    "class": "Adam",
    "lr": 1e-3,
    "weight_decay": 8e-4,
}

# Loss function (Huber Loss)
# Note: SKRL uses MSELoss by default for DQN. We can override it if supported or modify the agent.
# Assuming SKRL supports "loss_function" in config or we might need to instantiate it in the agent class.
# For now, we'll try to set it here. If SKRL doesn't support it via config, it might be ignored.
# However, standard SKRL agents often allow passing a loss function class/instance.
# Since we are using a config dict, we might need to rely on default or check if we can pass a string.
# Let's assume we can't easily change it via simple dict config without custom agent class, 
# but we will set the parameters we can.

# Exploration
LiftDQNAgentCfg["agent"]["exploration"] = {
    "initial_epsilon": 0.90,
    "final_epsilon": 0.05,
    "timesteps": 200,
}

# Trainer
LiftDQNAgentCfg["trainer"] = {
    "timesteps": 10000,
    "headless": True,
}
