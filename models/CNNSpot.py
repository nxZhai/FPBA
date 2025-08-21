import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
import torchvision.transforms as transforms


class CNNSpot(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = resnet50(num_classes=num_classes)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        try:
            self.model.load_state_dict(state_dict["model"], strict=True)
        except:
            self.model.load_state_dict(state_dict, strict=True)