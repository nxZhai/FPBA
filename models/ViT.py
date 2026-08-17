import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16


class ViT_B_16(nn.Module):
    def __init__(self, mode="eval", num_classes=1):
        super().__init__()

        if mode == "train":
            self.model = vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            )
            self.model.heads.head = nn.Linear(
                self.model.heads.head.in_features, num_classes
            )
        else:
            self.model = vit_b_16(weights=None, num_classes=num_classes)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def init_fc(self, init_gain=0.02):
        torch.nn.init.normal_(self.model.heads.head.weight.data, 0.0, init_gain)

    def forward(self, x):
        return self.model(self.normalize(x))

    def load_weights(self, opt):
        state_dict = torch.load(opt.ckpt, map_location="cpu")
        state_dict = state_dict.get("model", state_dict)
        self.model.load_state_dict(state_dict, strict=True)
