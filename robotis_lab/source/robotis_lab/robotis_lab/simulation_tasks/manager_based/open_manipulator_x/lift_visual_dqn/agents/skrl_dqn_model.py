import torch
import torch.nn as nn
import torchvision.models as models
from skrl.models.torch import Model, DeterministicMixin
import gymnasium as gym

class RGBD_DQN(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # [FIX] Detect ROI dimensions from observation space
        self.H = 69 # Default
        self.W = 34 # Default
        
        # Try to infer from observation space
        # Expecting Dict with "rgb" or "depth"
        if isinstance(observation_space, gym.spaces.Dict):
            # Check for "rgb"
            if "rgb" in observation_space.spaces:
                s = observation_space.spaces["rgb"].shape
                # (H, W, C) or (N, H, W, C)
                if len(s) == 3:
                    self.H, self.W = s[0], s[1]
                elif len(s) == 4:
                    self.H, self.W = s[1], s[2]
            elif "depth" in observation_space.spaces:
                s = observation_space.spaces["depth"].shape
                if len(s) == 3:
                    self.H, self.W = s[0], s[1]
                elif len(s) == 4:
                    self.H, self.W = s[1], s[2]
        
        print(f"[INFO] RGBD_DQN initialized with ROI: {self.W}x{self.H}")

        # Load MobileNetV2 backbones
        # We need the features, not the classifier
        # MobileNetV2 features output: (N, 1280, H/32, W/32) -> (N, 1280, 7, 7) for 224x224
        self.rgb_backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        self.depth_backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features

        # Combined features: 1280 + 1280 = 2560 channels
        self.input_channels = 2560
        
        # Decoder (Upsampling)
        # We need to go from (2560, 7, 7) -> (1, 224, 224)
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224 (5 upsamples)
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(self.input_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14 -> 28
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 28 -> 56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 56 -> 112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 112 -> 224
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
            # No activation at the end, raw Q-values
        )

    def compute(self, inputs, role):
        states = inputs["states"]
       
        if isinstance(states, dict):
            rgb = states.get("rgb")
            depth = states.get("depth")
        elif isinstance(states, torch.Tensor):
            # [FIX] Handle flattened tensor from Replay Memory
            # Skrl flattens Dict observations when storing in memory.
            # Keys are sorted alphabetically: "depth", "rgb"
            # Depth: (1, H, W)
            # RGB:   (3, H, W)
            
            N = states.shape[0]
            H, W = self.H, self.W
            depth_size = 1 * H * W
            rgb_size = 3 * H * W
            expected_size = depth_size + rgb_size
            
            if states.shape[1] == expected_size:
                # Split tensor
                # Sorted keys: "depth", "rgb" -> Depth comes first
                depth_flat = states[:, :depth_size]
                rgb_flat = states[:, depth_size:]
                
                # Reshape
                depth = depth_flat.view(N, 1, H, W)
                rgb = rgb_flat.view(N, 3, H, W)
            else:
                raise ValueError(f"RGBD_DQN received tensor with shape {states.shape}, expected flattened size {expected_size} (Depth+RGB for {W}x{H}).")
        else:
            raise ValueError(f"RGBD_DQN model expects a dictionary observation or flattened tensor. Received: {type(states)}")

        # Handle RGB (N, H, W, C) -> (N, C, H, W)
        if rgb.dim() == 4:
            if rgb.shape[-1] == 4: # RGBA
                rgb = rgb.permute(0, 3, 1, 2)
                rgb = rgb[:, :3, :, :]
            elif rgb.shape[-1] == 3: # RGB
                rgb = rgb.permute(0, 3, 1, 2)
        
        # Handle Depth (N, H, W, 1) -> (N, 1, H, W)
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2)
           
        # Normalize RGB
        rgb = rgb.float()
        # Safety check: only normalize if values are in [0, 255] range
        if rgb.max() > 1.1:
            rgb = rgb / 255.0
            
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        rgb = (rgb - mean) / std

        # Repeat depth to 3 channels
        depth = depth.repeat(1, 3, 1, 1)
       
        # Extract features
        # (N, 3, 224, 224) -> (N, 1280, 7, 7)
        rgb_features = self.rgb_backbone(rgb)
        depth_features = self.depth_backbone(depth)
       
        # Concatenate
        # (N, 2560, 7, 7)
        combined_features = torch.cat([rgb_features, depth_features], dim=1)
       
        # Decode to Q-Map
        # (N, 1, 224, 224)
        q_map = self.decoder(combined_features)
        
        # [FIX] Force resize to (H, W) to match action space
        if q_map.shape[-2:] != (self.H, self.W):
             q_map = torch.nn.functional.interpolate(q_map, size=(self.H, self.W), mode='bilinear', align_corners=False)
        
        # Flatten to (N, H*W)
        q_values = q_map.view(q_map.size(0), -1)
        
        # [DEBUG] Print greedy action info (only occasionally to avoid spam)
        # We can't easily access epsilon here, but we know this is the greedy path
        if torch.rand(1).item() < 0.01: # 1% chance to print
             print(f"[DEBUG] Greedy Act (compute): Max Q-value: {q_values.max().item():.4f}")
       
        return q_values, {}

    def random_act(self, inputs, role):
        """
        Override random_act to return random pixel indices.
        """
        # We need to know the batch size
        # inputs["states"] is a dict
        states = inputs["states"]
        if isinstance(states, dict):
            # Check any key to get batch size
            batch_size = next(iter(states.values())).shape[0]
        else:
            # Fallback if states is tensor (should not happen with this model)
            batch_size = states.shape[0]
            
        # Generate random indices in range [0, num_actions)
        # num_actions is 224*224 = 50176
        # We can get it from action_space if available, or hardcode/infer
        if self.action_space is not None:
            if hasattr(self.action_space, "n"):
                num_actions = self.action_space.n
            else:
                num_actions = self.H * self.W # Fallback
        else:
            num_actions = self.H * self.W
            
        random_actions = torch.randint(0, num_actions, (batch_size, 1), device=self.device)
        
        # [DEBUG] Print random action info
        if torch.rand(1).item() < 0.01: # 1% chance to print
             print(f"[DEBUG] Random Act: Generated {batch_size} candidates (SKRL will use these if epsilon check passes)")
             
        return random_actions, None, None
