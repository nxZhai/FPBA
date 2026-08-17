import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50


class CNNSpot(nn.Module):
    def __init__(self, mode="eval", num_classes=1):
        super().__init__()

        if mode == "train":
            self.model = resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = resnet50(weights=None, num_classes=num_classes)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, init_gain)

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        self.model.load_state_dict(state_dict, strict=True)
