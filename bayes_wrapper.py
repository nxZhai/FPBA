import os
import torch
import torch.nn as nn

from models import get_model


class BayesWrapper(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self._check()

        self.classifier = get_model(opt)
        self.classifier.load_weights(opt)
        self.separate_fc()

        self.classifier_parallel()

        self.appended_model_list = [self.create_mlp() 
                                    for _ in range(self.opt.bayes_model_num)]

        print(f"Load {len(self.appended_model_list)} append models.")

        for idx, mlp in enumerate(self.appended_model_list):
            self.load_weights(model_No=idx)
            mlp = nn.DataParallel(mlp)
            if self.opt.mode == 'train':
                mlp.train()
            else:
                mlp.eval()
            mlp.cuda()

    def _check(self):
        assert self.opt.model in ['CNNSpot', 'MobileNet']
        assert self.opt.bayes

    def separate_fc(self):
        if self.opt.model == 'CNNSpot':
            self.in_features = self.classifier.model.fc.in_features
            self.cls_layer = self.classifier.model.fc
            self.classifier.model.fc = nn.Sequential()
        elif self.opt.model == 'MobileNet':
            self.in_features = self.classifier.model.classifier[-1].in_features
            self.cls_layer = self.classifier.model.classifier[-1]
            self.classifier.model.classifier[-1] = nn.Sequential()

    def classifier_parallel(self):
        self.classifier = nn.DataParallel(self.classifier)
        self.cls_layer.eval().cuda()

        self.cls_layer = nn.DataParallel(self.cls_layer)
        self.classifier.eval().cuda()

    def create_mlp(self):
        appended_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.in_features),
            torch.nn.ReLU(),
            torch.nn.Linear(self.in_features, 1))
        return appended_mlp

    def forward(self, x, model_No=-1):
        pred = torch.zeros((len(x),1), dtype=torch.float).cuda()
        fea = self.classifier(x)
        x_out = self.cls_layer(fea)
        
        if model_No == -1:
            for i in range(len(self.appended_model_list)):
                pred += (x_out + self.appended_model_list[i](fea))
                
            return pred / self.opt.bayes_model_num
        else:
            return (x_out + self.appended_model_list[model_No](fea))

    def load_weights(self, model_No=-1):
        if model_No == -1:
            for i in range(self.opt.bayes_model_num):
                
                weight_path = os.path.join(self.opt.appmodel_ckpt_root,
                                            str(i) + self.opt.appmodel_ckpt_name)
                state_dict = torch.load(weight_path, map_location='cpu')
                try:
                    self.appended_model_list[i].load_state_dict(state_dict, strict=True)
                except Exception as error:
                    raise ValueError(f'[ERROR] meet error when load weights in {weight_path}.')
        else:
            weight_path = os.path.join(self.opt.appmodel_ckpt_root,
                            str(model_No) + self.opt.appmodel_ckpt_name)
            state_dict = torch.load(weight_path, map_location='cpu')
            try:
                self.appended_model_list[model_No].load_state_dict(state_dict, strict=True)
            except Exception as error:
                raise ValueError(f'[ERROR] meet error when load weights in {weight_path}.')

    def train(self, model_No=-1):
        if model_No == -1:
            for mlp in self.appended_model_list:
                mlp.train().cuda()
        else:
            self.appended_model_list[model_No].train().cuda()

    def eval(self, model_No=-1):
        if model_No == -1:
            for mlp in self.appended_model_list:
                mlp.eval().cuda()
        else:
            self.appended_model_list[model_No].eval().cuda()