import torch
import torch.nn as nn
import torch
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
        if self.opt.model == 'LGrad':
            images.requires_grad = True
        return self.model(images)

    def criterion(self, output, labels):
        if self.opt.targeted:
            cost = -nn.BCEWithLogitsLoss()(output.squeeze(1), labels.float())
        else:
            cost = nn.BCEWithLogitsLoss()(output.squeeze(1), labels.float())

        return cost


class FPBA(BaseAttack):
    """
    Our method in MM24 unpublished paper.
    """ 
    def __init__(self, opt, model):
        super().__init__(opt, model)
        self.rho = self.opt.rho
        self.N = self.opt.N

    def frequency_attack(self, x, labels):
        x = x.clone().detach()
        noise = 0
        
        for n in range(self.N):
            x = x + torch.empty_like(x).uniform_(-self.opt.epsilon, self.opt.epsilon)
            x = torch.clamp(x, min=0, max=1).detach()
            x_dct = dct_2d(x).cuda()
            mask = (torch.rand_like(x) * 2 * self.rho + 1 - self.rho).cuda()
            x_idct = idct_2d(x_dct * mask)

            x_idct.requires_grad = True
            output = self.pred(x_idct)
            # calculate
            cost = self.criterion(output, labels)
            grad = torch.autograd.grad(
                    cost, x_idct, retain_graph=False, create_graph=False
                    )[0]
            noise += grad
            
        return noise/self.N

    def spatial_attack(self, x, labels):
        x = x.clone().detach()
        x.requires_grad = True
        output = self.pred(x)
        cost = self.criterion(output, labels)
        
        grad = torch.autograd.grad(
                cost, x, retain_graph=False, create_graph=False
                )[0]
        
        return grad

    def attack(self, images, labels):
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        if self.opt.targeted:
            labels = (labels+1)%2

        ori_images = images.clone().detach()

        for i in range(self.opt.steps):
            grad = torch.zeros_like(images).cuda()
            
            freq_grad = self.frequency_attack(images, labels)
            grad += freq_grad
            spat_grad = self.spatial_attack(images, labels)
            grad += spat_grad
            
            images = images + self.opt.alpha * torch.sign(grad)
            delta = torch.clamp(images - ori_images,
                        min = -self.opt.epsilon,
                        max = self.opt.epsilon)
            images = torch.clamp(ori_images + delta, min=0, max=1).detach()

        return images