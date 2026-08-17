import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet34


class Spec(nn.Module):
    def __init__(self, mode="eval", num_classes=1):
        super().__init__()

        if mode == "train":
            self.classifier = resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        else:
            self.classifier = resnet34(weights=None, num_classes=num_classes)

    def forward(self, x):
        x = torch.fft.fft2(x, dim=(-2, -1))
        x = torch.log(torch.abs(x) + 1e-3)
        x_min = x.flatten(-2, -1).quantile(q=0.05, dim=-1).unsqueeze(-1).unsqueeze(-1)
        x_max = x.flatten(-2, -1).quantile(q=0.95, dim=-1).unsqueeze(-1).unsqueeze(-1)
        x = (x - x_min) / (x_max - x_min)
        x = (x - 0.5) * 2
        return self.classifier(x.clamp(-1.0, 1.0))

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.classifier.fc.weight.data, 0.0, init_gain)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        self.classifier.load_state_dict(state_dict, strict=True)
