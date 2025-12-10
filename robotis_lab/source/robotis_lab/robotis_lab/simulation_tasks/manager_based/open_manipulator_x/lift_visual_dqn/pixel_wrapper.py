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
        import traceback
        print("PixelActionWrapper initialized by:")
        traceback.print_stack()
        super().__init__(env)
        self.env = env
        
        # ROI Definition (x, y, w, h)
        self.roi_x = 100
        self.roi_y = 70
        self.roi_w = 34
        self.roi_h = 69
        
        # Full Image dimensions
        self.full_height = camera_cfg.height
        self.full_width = camera_cfg.width
        
        # Cropped Dimensions (Action/Obs Space)
        self.height = self.roi_h
        self.width = self.roi_w
        self.num_actions = self.height * self.width
        
        # Override action space
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
        # Camera Intrinsics (Based on FULL image)
        # f_x = W * f / w_a
        self.focal_length = camera_cfg.spawn.focal_length
        self.horiz_aperture = camera_cfg.spawn.horizontal_aperture
        
        self.fx = self.full_width * self.focal_length / self.horiz_aperture
        self.fy = self.fx # Square pixels
        self.cx = self.full_width / 2.0
        self.cy = self.full_height / 2.0
        
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
        
        # Update Observation Space to match ROI
        self._update_observation_space()

    def _update_observation_space(self):
        """Updates the observation space to reflect the cropped dimensions and filter keys."""
        # Deep copy the original observation space to avoid modifying the original env's space if shared
        # We'll reconstruct the relevant parts.
        
        original_space = self.env.observation_space
        if isinstance(original_space, gym.spaces.Dict) and "policy" in original_space.spaces:
            policy_space = original_space.spaces["policy"]
            if isinstance(policy_space, gym.spaces.Dict):
                new_policy_spaces = {}
                # [FIX] Only keep RGB and Depth to ensure clean flattened tensor in SKRL
                # This avoids issues with other keys (proprioception, etc.) messing up the splitting logic
                target_keys = ["rgb", "depth"]
                
                for key in target_keys:
                    if key in policy_space.spaces:
                        space = policy_space.spaces[key]
                        # Update shape: (H, W, C) -> (roi_h, roi_w, C)
                        if isinstance(space, gym.spaces.Box):
                            low = space.low
                            high = space.high
                            dtype = space.dtype
                            
                            # Handle shape
                            # If shape is (H, W, C)
                            if len(space.shape) == 3:
                                channels = space.shape[2]
                                new_shape = (self.roi_h, self.roi_w, channels)
                                
                                # Create new Box
                                new_space = gym.spaces.Box(low=0, high=255 if key == "rgb" else float('inf'), shape=new_shape, dtype=dtype)
                                new_policy_spaces[key] = new_space
                            # If shape is (N, H, W, C) - Vectorized Env
                            elif len(space.shape) == 4:
                                N = space.shape[0]
                                channels = space.shape[3]
                                new_shape = (N, self.roi_h, self.roi_w, channels)
                                
                                # Create new Box
                                new_space = gym.spaces.Box(low=0, high=255 if key == "rgb" else float('inf'), shape=new_shape, dtype=dtype)
                                new_policy_spaces[key] = new_space
                            else:
                                new_policy_spaces[key] = space
                
                # Replace the policy space
                new_policy_space = gym.spaces.Dict(new_policy_spaces)
                
                # Reconstruct the top-level Dict
                # We only keep "policy" to be safe
                self.observation_space = gym.spaces.Dict({"policy": new_policy_space})
                
                # [HACK] Force update the unwrapped environment's observation space
                if hasattr(self.env, "unwrapped"):
                    self.env.unwrapped.observation_space = self.observation_space
                    print(f"[INFO] PixelActionWrapper: Forced update of unwrapped env observation space.")
                
                print(f"[INFO] PixelActionWrapper: Updated observation space to {self.roi_w}x{self.roi_h} (Keys: {list(new_policy_spaces.keys())})")

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._crop_observation(obs)
        return obs, info

    def _crop_observation(self, obs):
        """Crops the observation images to the ROI and filters keys."""
        if "policy" in obs:
            # Filter keys in policy
            new_policy = {}
            for key in ["depth", "rgb"]:
                if key in obs["policy"]:
                    val = obs["policy"][key]
                    # Assume (N, H, W, C)
                    if isinstance(val, torch.Tensor) and val.ndim == 4:
                        new_policy[key] = val[:, self.roi_y:self.roi_y+self.roi_h, self.roi_x:self.roi_x+self.roi_w, :]
                    else:
                        new_policy[key] = val
            
            # Replace policy dict with filtered one
            obs["policy"] = new_policy
            
            # Remove other top-level keys if necessary? 
            # SKRL might expect the exact structure of observation_space.
            # Since we defined observation_space as {"policy": ...}, we should probably only return that.
            # But returning extra keys usually doesn't hurt unless SKRL flattens them.
            # To be safe, let's filter the top level too.
            keys_to_keep = ["policy"]
            obs = {k: v for k, v in obs.items() if k in keys_to_keep}
            
        return obs
        
    def step(self, action_id):
        """
        Args:
            action_id: Tensor containing the pixel index (0..N-1) of shape (num_envs,)
                       where N = roi_w * roi_h
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
            
        # 1. Convert Action ID to (u, v) in ROI Frame
        # action_id shape: (num_envs,)
        # Clamp action_id to valid range to prevent index out of bounds
        action_id = torch.clamp(action_id, 0, self.num_actions - 1)
        
        u_crop = action_id % self.width
        v_crop = action_id // self.width
        
        # Convert to Full Image Frame
        u = u_crop + self.roi_x
        v = v_crop + self.roi_y
        
        # 2. Get Depth from current observation (FULL IMAGE)
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
        
        # [LOG] Predicted XYZ
        if num_envs > 0:
            # Print with 4 decimal places for clarity
            xyz = p_world[0].tolist()
            uv = (u[0].item(), v[0].item())
            depth_val = d[0].item()
            cam_pt = p_cam[0].tolist()
            print(f"[INFO] Env 0: Pixel (Full): {uv}, Depth: {depth_val:.4f}")
            print(f"       Cam: [{cam_pt[0]:.4f}, {cam_pt[1]:.4f}, {cam_pt[2]:.4f}]")
            print(f"       World: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")

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
        
        # Crop the final observation before returning
        obs = self._crop_observation(obs)
        
        return obs, total_reward, terminated, truncated, info

