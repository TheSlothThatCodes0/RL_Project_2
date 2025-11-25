
import argparse
import sys
import os
import torch

# Add source to path to ensure robotis_lab is importable
# Based on structure: robotis_lab/source/robotis_lab contains the robotis_lab package
sys.path.append("/home/pi0/RL_GOOD/robotis_lab/source/robotis_lab")

from isaaclab.app import AppLauncher

# Create the app launcher
parser = argparse.ArgumentParser(description="Debug Gripper Closing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from robotis_lab.simulation_tasks.manager_based.open_manipulator_x.lift.joint_pos_env_cfg import OpenManipulatorXCubeLiftEnvCfg
from isaaclab.managers import SceneEntityCfg

def main():
    print("[INFO] Configuring Environment...")
    # 1. Configure Environment
    env_cfg = OpenManipulatorXCubeLiftEnvCfg()
    
    # Set number of environments to 1
    env_cfg.scene.num_envs = 1
    
    # Disable randomization
    env_cfg.observations.policy.enable_corruption = False
    
    # Configure Rewards: Only incentivize gripper closing
    # We modify the existing object_grasped term to ignore distance
    if hasattr(env_cfg.rewards, "object_grasped"):
        print("[INFO] Modifying object_grasped reward to ignore distance (focus on closing)...")
        # Set huge thresholds so distance condition is always true
        env_cfg.rewards.object_grasped.params["xy_diff_threshold"] = 100.0
        env_cfg.rewards.object_grasped.params["z_diff_threshold"] = 100.0
        # Keep gripper thresholds as defined in config to test them
    
    # Disable other rewards
    for term_name in ["reaching_object", "lifting_object", "object_goal_tracking", 
                      "object_goal_tracking_fine_grained", "action_rate", "joint_vel"]:
        if hasattr(env_cfg.rewards, term_name):
            setattr(env_cfg.rewards, term_name, None)

    # Disable terminations
    env_cfg.terminations = None
    
    # Disable curriculum
    env_cfg.curriculum = None

    # 2. Create Environment
    print("[INFO] Creating Environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 3. Run Loop
    print("[INFO] Starting simulation loop...")
    env.reset()
    
    # Get robot entity
    robot = env.scene["robot"]
    print(f"[INFO] Robot Joint Names: {robot.joint_names}")
    
    # Indices for gripper joints
    gripper_indices = [i for i, n in enumerate(robot.joint_names) if "gripper" in n]
    print(f"[INFO] Gripper Joint Indices: {gripper_indices}")

    # Simulation parameters
    dt = 0.02 # Control dt (decimation=2, sim_dt=0.01)
    hold_duration = 5.0 # seconds
    steps_per_phase = int(hold_duration / dt)
    total_steps = steps_per_phase * 2
    
    step_count = 0
    while simulation_app.is_running():
        # Create actions
        # 4 arm joints + 1 gripper action (binary)
        actions = torch.zeros((env.num_envs, 5), device=env.device)
        
        # Determine phase
        if step_count < steps_per_phase:
            phase = "Closing/Holding Closed"
            actions[:, 4] = -1.0 # Close command
        else:
            phase = "Opening/Holding Open"
            actions[:, 4] = 1.0 # Open command
            
        # Step environment
        obs, rew, terminated, truncated, extras = env.step(actions)
        
        # Print Debug Info
        if step_count % 10 == 0: # Print every 10 steps to reduce clutter
            print(f"\nStep {step_count} ({phase}):")
            print(f"  Action Sent: {actions[0].cpu().numpy()}")
            
            joint_pos = robot.data.joint_pos[0]
            # print(f"  Robot Joint Pos: {joint_pos.cpu().numpy()}")
            
            for idx in gripper_indices:
                name = robot.joint_names[idx]
                val = joint_pos[idx].item()
                print(f"    {name}: {val:.6f}")
                
            print(f"  Reward: {rew[0].item():.6f}")
        
        step_count += 1
        if step_count >= total_steps:
            break

    print("[INFO] Closing environment...")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
