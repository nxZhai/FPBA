import torch
import torch.nn as nn
from torch.autograd import Variable as V

from utils import dct_2d, idct_2d


class BaseAttack:
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model

    def attack(self, images, labels):
        return NotImplementedError("Not Implemented Error.")

    def pred(self, images):
        if self.opt.earlystop:
            assert self.model.training == False
        if self.opt.model == "LGrad":
            images.requires_grad = True
        return self.model(images)

    def criterion(self, output, labels):
        if self.opt.targeted:
            cost = -nn.BCEWithLogitsLoss()(output.squeeze(1), labels.float())
        else:
            cost = nn.BCEWithLogitsLoss()(output.squeeze(1), labels.float())

        return cost

    def _model_gradient(self, x, labels, model_no=None):
        if model_no is None:
            output = self.pred(x)
        else:
            output = self.model.forward_logits(x, model_No=model_no)
        cost = self.criterion(output, labels)
        return torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

    def gradient(self, x, labels):
        if not self.opt.bayes:
            return self._model_gradient(x, labels)

        gradients = [
            self._model_gradient(x, labels, model_no=model_no)
            for model_no in range(self.opt.bayes_model_num)
        ]
        return torch.stack(gradients, dim=0).mean(dim=0)


class FPBA(BaseAttack):
    def __init__(self, opt, model):
        super().__init__(opt, model)
        self.rho = self.opt.rho
        self.N = self.opt.N

    def frequency_attack(self, x, labels):
        x = x.clone().detach()
        noise = 0

        for n in range(self.N):
            gauss = torch.randn_like(x) * self.opt.epsilon
            x_noisy = torch.clamp(x + gauss, min=0, max=1).detach()
            x_dct = dct_2d(x_noisy).cuda()
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
            x_idct = idct_2d(x_dct * mask)

            x_idct.requires_grad = True
            grad = self.gradient(x_idct, labels)
            noise += grad

        return noise / self.N

    def spatial_attack(self, x, labels):
        x = x.clone().detach()
        x.requires_grad = True
        return self.gradient(x, labels)

    def attack(self, images, labels):
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        if self.opt.targeted:
            labels = (labels + 1) % 2

        ori_images = images.clone().detach()

        for i in range(self.opt.steps):
            grad = torch.zeros_like(images).cuda()

            freq_grad = self.frequency_attack(images, labels)
            grad += freq_grad
            spat_grad = self.spatial_attack(images, labels)
            grad += spat_grad

            images = images + self.opt.alpha * torch.sign(grad)
            delta = torch.clamp(
                images - ori_images, min=-self.opt.epsilon, max=self.opt.epsilon
            )
            images = torch.clamp(ori_images + delta, min=0, max=1).detach()

        return images
