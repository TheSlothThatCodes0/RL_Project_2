import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from scipy.spatial.transform import Rotation
import numpy as np

class PixelActionWrapper(gym.Wrapper):
    """
    Wrapper that converts a discrete pixel action (0..H*W-1) into a 3D grasp target
    and executes it using Inverse Kinematics (Macro-Step).
    """
    def __init__(self, env: ManagerBasedRLEnv, camera_cfg):
        super().__init__(env)
        self.env = env
        
        # Image dimensions
        self.height = camera_cfg.height
        self.width = camera_cfg.width
        self.num_actions = self.height * self.width
        
        # Override action space
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
        # Camera Intrinsics
        # f_x = W * f / w_a
        self.focal_length = camera_cfg.spawn.focal_length
        self.horiz_aperture = camera_cfg.spawn.horizontal_aperture
        
        self.fx = self.width * self.focal_length / self.horiz_aperture
        self.fy = self.fx # Square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # Camera Extrinsics
        # Config: pos=(1.4, 0.05, 0.4), rot=(0.54, 0.41, 0.44, 0.57) (w, x, y, z)
        # Note: Scipy uses (x, y, z, w)
        pos = camera_cfg.offset.pos
        rot = camera_cfg.offset.rot # (w, x, y, z)
        
        self.cam_pos = torch.tensor(pos, device=self.env.device, dtype=torch.float32)
        
        # Convert (w, x, y, z) -> (x, y, z, w) for Scipy
        rot_scipy = [rot[1], rot[2], rot[3], rot[0]]
        r = Rotation.from_quat(rot_scipy)
        self.R_cam_to_world = torch.tensor(r.as_matrix(), device=self.env.device, dtype=torch.float32)
        
        # Pre-calculate pixel grid for batch processing if needed
        # For now, we process single action
        
    def step(self, action_id):
        """
        Args:
            action_id: Tensor containing the pixel index (0..50175) of shape (num_envs,)
        """
        # Ensure action_id is a tensor
        if not isinstance(action_id, torch.Tensor):
            action_id = torch.tensor(action_id, device=self.env.device)
        
        # Ensure action_id is long for indexing
        action_id = action_id.long()
        
        # Handle shape mismatch (e.g. if skrl passes (N, N) instead of (N, 1))
        # We expect one action per environment
        num_envs = self.env.num_envs
        
        if action_id.numel() > num_envs:
            # If we have more actions than envs, try to take the first column or slice
            if action_id.dim() > 1 and action_id.shape[0] == num_envs:
                action_id = action_id[:, 0]
            else:
                action_id = action_id.flatten()[:num_envs]
        elif action_id.numel() < num_envs:
             # Should not happen, but handle it
             pass
             
        # Flatten to (N,)
        action_id = action_id.flatten()
            
        # 1. Convert Action ID to (u, v)
        # action_id shape: (num_envs,)
        # Clamp action_id to valid range to prevent index out of bounds
        action_id = torch.clamp(action_id, 0, self.num_actions - 1)
        
        u = action_id % self.width
        v = action_id // self.width
        
        # 2. Get Depth from current observation
        obs_dict = self.env.observation_manager.compute()
        # Shape: (num_envs, H, W, 1)
        depth_img = obs_dict["policy"]["depth"] 
        
        # Get depth value for each env
        # depth_img: (N, H, W, 1)
        # We want d[i] = depth_img[i, v[i], u[i], 0]
        
        num_envs = depth_img.shape[0]
        batch_indices = torch.arange(num_envs, device=self.env.device, dtype=torch.long)
        
        if depth_img.shape[-1] == 1:
            # (N, H, W, 1)
            d = depth_img[batch_indices, v, u, 0]
        else:
            # Assume (N, 1, H, W)
            d = depth_img[batch_indices, 0, v, u]
        
        # Clip depth to valid range
        d = torch.clamp(d, 0.01, 2.0)
        
        # 3. Deproject to Camera Frame
        # Vectorized calculation
        x_cam = (u - self.cx) * d / self.fx
        y_cam = (self.cy - v) * d / self.fy
        z_cam = -d
        
        # Stack to create p_cam: (N, 3)
        p_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)
        
        # 4. Transform to World Frame
        # p_world = (R @ p_cam.T).T + pos
        # R is (3, 3), p_cam is (N, 3) -> (3, N)
        p_world = torch.matmul(self.R_cam_to_world, p_cam.T).T + self.cam_pos
        
        # HARDCODED Z-HEIGHT FIX
        # Force Z to be slightly above the table (e.g., 0.025m) to ensure grasping
        # The table is at z=0 (GroundPlane at -1.05, but Table top is usually at 0 in this setup? 
        # Wait, GroundPlane is at -1.05. Table is usually 1.05m high?
        # Let's check ObjectTableSceneCfg.
        # Usually table top is at z=0 in the local env frame if the robot is mounted on it.
        # Let's assume z=0.025 is a good grasp height relative to robot base.
        p_world[:, 2] = 0.025
        
        # 5. Execute Macro-Step
        # Target Orientation: Pointing down (0, 0, 1, 0) - (w, x, y, z) ?
        # DifferentialIKControllerCfg uses (w, x, y, z) for quat if command_type="pose"
        # Let's assume (0, 1, 0, 0) or similar. 
        # If we use (0, 0, 1, 0) -> 180 deg rotation around Y?
        # Let's use a fixed orientation for now.
        target_quat = torch.tensor([0, 0, 1, 0], device=self.env.device).repeat(num_envs, 1)
        
        # Construct Action: (pos, rot) -> (N, 7)
        target_pose = torch.cat([p_world, target_quat], dim=1)
        
        # Gripper Open (1.0) -> (N, 1)
        gripper_open = torch.ones((num_envs, 1), device=self.env.device)
        
        # Combine Arm + Gripper -> (N, 8)
        full_action = torch.cat([target_pose, gripper_open], dim=1)
        
        # Loop to reach target
        total_reward = torch.zeros(num_envs, device=self.env.device)
        
        # Move to target
        for _ in range(40):
            obs, rew, terminated, truncated, info = self.env.step(full_action)
            total_reward += rew
            # Note: We don't break early in vectorized envs usually, unless all are done.
            # But here we are running a macro-step.
            # If an env terminates, we should probably stop sending actions to it?
            # For simplicity, we continue sending actions (reset envs will ignore or handle it).
                
        # Close Gripper
        gripper_close = -1.0 * torch.ones((num_envs, 1), device=self.env.device)
        full_action_close = torch.cat([target_pose, gripper_close], dim=1)
        
        for _ in range(20):
            obs, rew, terminated, truncated, info = self.env.step(full_action_close)
            total_reward += rew
                    
        # Lift (Optional)
        lift_target = p_world + torch.tensor([0, 0, 0.2], device=self.env.device)
        lift_pose = torch.cat([lift_target, target_quat], dim=1)
        lift_action = torch.cat([lift_pose, gripper_close], dim=1)
        
        for _ in range(20):
            obs, rew, terminated, truncated, info = self.env.step(lift_action)
            total_reward += rew
            
        # DEBUG: Print mean reward
        if num_envs > 0:
            print(f"[Mean] Macro-Step Reward: {total_reward.mean().item()}")
                    
        return obs, total_reward, terminated, truncated, info

