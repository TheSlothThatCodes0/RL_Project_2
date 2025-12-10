import torch
import torch.nn as nn
import torchvision.models as models
from skrl.models.torch import Model, DeterministicMixin

class RGBD_DQN(Model, DeterministicMixin):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

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
        else:
            raise ValueError("RGBD_DQN model expects a dictionary observation with 'rgb' and 'depth' keys.")

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
        
        # Flatten to (N, 224*224)
        q_values = q_map.view(q_map.size(0), -1)
       
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
                num_actions = 50176 # Fallback
        else:
            num_actions = 50176
            
        random_actions = torch.randint(0, num_actions, (batch_size, 1), device=self.device)
        return random_actions, None, None
