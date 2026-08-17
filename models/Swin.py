import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.models import swin_b


class Swin_B(nn.Module):
    def __init__(self, mode="eval", num_classes=1):
        super().__init__()

        if mode == "train":
            self.model = swin_b(weights=torchvision.models.Swin_B_Weights.IMAGENET1K_V1)
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        else:
            self.model = swin_b(weights=None, num_classes=num_classes)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        self.model.load_state_dict(state_dict, strict=True)
