import torch
import torch.nn as nn
from torchvision.models import swin_b
from torchvision import transforms


class Swin_B(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()
        
        self.model = swin_b(num_classes=num_classes)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        try:
            self.model.load_state_dict(state_dict["model"], strict=True)
        except:
            self.model.load_state_dict(state_dict, strict=True)