import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.mobilenet import mobilenet_v2


class MobileNet(nn.Module):
    def __init__(self, mode="eval", num_classes=1):
        super().__init__()

        if mode == "train":
            self.model = mobilenet_v2(
                weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
            )
            self.model.classifier[-1] = nn.Linear(
                self.model.classifier[-1].in_features, num_classes
            )
        else:
            self.model = mobilenet_v2(weights=None, num_classes=num_classes)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.model.classifier[-1].weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(self.normalize(x))

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        self.model.load_state_dict(state_dict, strict=True)
