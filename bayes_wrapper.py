import os

import torch
import torch.nn as nn

from models import get_model


class BayesWrapper(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self._check()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt.appmodel_ckpt_root = self.opt.appmodel_ckpt_root or os.path.join(
            self.opt.checkpoints_dir, "appended_mlp", self.opt.model
        )
        self.opt.appmodel_ckpt_name = self.opt.appmodel_ckpt_name or (
            f"_{self.opt.model}_PYX_AppendedModel_AT.pth"
        )

        self.classifier = get_model(opt)
        self.classifier.load_weights(opt)
        self.separate_fc()

        self.classifier_parallel()

        self.appended_model_list = nn.ModuleList(
            [self.create_mlp() for _ in range(self.opt.bayes_model_num)]
        )

        print(f"Load {len(self.appended_model_list)} append models.")

        for idx, mlp in enumerate(self.appended_model_list):
            if self.opt.mode != "train":
                self.load_weights(model_No=idx)
            mlp.to(self.device)

        self.eval() if self.opt.mode != "train" else self.train()

    def _check(self):
        assert self.opt.model in ["CNNSpot", "MobileNet"]
        assert self.opt.bayes

    def separate_fc(self):
        if self.opt.model == "CNNSpot":
            self.in_features = self.classifier.model.fc.in_features
            self.cls_layer = self.classifier.model.fc
            self.classifier.model.fc = nn.Sequential()
        elif self.opt.model == "MobileNet":
            self.in_features = self.classifier.model.classifier[-1].in_features
            self.cls_layer = self.classifier.model.classifier[-1]
            self.classifier.model.classifier[-1] = nn.Sequential()

    def classifier_parallel(self):
        if self.device.type == "cuda":
            self.classifier = nn.DataParallel(self.classifier)
            self.cls_layer = nn.DataParallel(self.cls_layer)
        self.cls_layer.eval().to(self.device)
        self.classifier.eval().to(self.device)

    def create_mlp(self):
        appended_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_features, 1),
        )
        return appended_mlp

    def _forward_features(self, x):
        fea = self.classifier(x)
        x_out = self.cls_layer(fea)
        return fea, x_out

    def forward_logits(self, x, model_No):
        fea, x_out = self._forward_features(x)
        return x_out + self.appended_model_list[model_No](fea)

    def predict_proba(self, x):
        fea, x_out = self._forward_features(x)
        logits = torch.stack(
            [x_out + mlp(fea) for mlp in self.appended_model_list],
            dim=0,
        )
        return logits.sigmoid().mean(dim=0)

    def forward(self, x, model_No=None):
        if model_No is None:
            return self.predict_proba(x)
        return self.forward_logits(x, model_No=model_No)

    def load_weights(self, model_No=-1):
        if model_No == -1:
            for i in range(self.opt.bayes_model_num):
                weight_path = os.path.join(
                    self.opt.appmodel_ckpt_root, str(i) + self.opt.appmodel_ckpt_name
                )
                state_dict = torch.load(weight_path, map_location="cpu")
                try:
                    self.appended_model_list[i].load_state_dict(state_dict, strict=True)
                except Exception as error:
                    raise ValueError(
                        f"[ERROR] meet error when load weights in {weight_path}."
                    )
        else:
            weight_path = os.path.join(
                self.opt.appmodel_ckpt_root, str(model_No) + self.opt.appmodel_ckpt_name
            )
            state_dict = torch.load(weight_path, map_location="cpu")
            try:
                self.appended_model_list[model_No].load_state_dict(
                    state_dict, strict=True
                )
            except Exception as error:
                raise ValueError(
                    f"[ERROR] meet error when load weights in {weight_path}."
                )

    def train(self, mode=True, model_No=-1):
        super().train(mode)
        self.classifier.eval()
        self.cls_layer.eval()
        if model_No == -1:
            models = self.appended_model_list
        else:
            models = [self.appended_model_list[model_No]]
        for mlp in models:
            mlp.train(mode).to(self.device)
        return self

    def eval(self, model_No=-1):
        return self.train(False, model_No=model_No)
