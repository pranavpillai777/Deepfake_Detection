import torch
import torchvision.models as models
import torch.nn as nn

# Initialize model architecture matching the checkpoint dimensions
model = models.mobilenet_v2(weights=None)
model.features[0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

checkpoint_path = r"C:\Users\Toshiba\Desktop\LY_PROJECT\checkpoints\mobilenet_v2_deepfake.pth"

# Load checkpoint and weights successfully
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

print("Model setup complete and weights loaded successfully!")