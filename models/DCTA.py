import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet50

from utils import dct_2d


class DCTA(nn.Module):
    def __init__(self, mode="eval", mean=None, var=None, num_classes=1, log=True):
        super().__init__()
        if mean is None or var is None:
            raise ValueError("DCTA requires --dcta_ckpt_dir with mean and var files.")

        dct_mean = torch.load(mean, map_location="cpu").float()
        dct_var = torch.load(var, map_location="cpu").float()
        if dct_mean.ndim == 3:
            dct_mean = dct_mean.unsqueeze(0)
        if dct_var.ndim == 3:
            dct_var = dct_var.unsqueeze(0)
        self.register_buffer("dct_mean", dct_mean)
        self.register_buffer("dct_var", dct_var)
        self.log = log
        self.epsilon = 1e-12

        if mode == "train":
            self.classifier = resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        else:
            self.classifier = resnet50(weights=None, num_classes=num_classes)

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.classifier.fc.weight.data, 0.0, init_gain)

    def forward(self, x):
        x = dct_2d(x * 255.0, norm="ortho")
        if self.log:
            x = torch.log(torch.abs(x) + self.epsilon)
        x = (x - self.dct_mean) / torch.sqrt(self.dct_var)
        return self.classifier(x)

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        if "netC" in state_dict:
            state_dict = state_dict["netC"]
        else:
            state_dict = state_dict.get("model", state_dict)
        self.classifier.load_state_dict(state_dict, strict=True)
