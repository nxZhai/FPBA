import os

import numpy as np
import torch
import torch.nn as nn

from bayes_wrapper import BayesWrapper
from optimizers.SGHMC import SGAdaHMC


class BayesTrainer:
    def __init__(self, opt, val_opt=None):
        if opt.model not in {"CNNSpot", "MobileNet"}:
            raise ValueError("BayesTrainer supports CNNSpot and MobileNet only.")
        if not opt.ckpt:
            raise ValueError("Bayes training requires --ckpt for the detector.")

        self.opt = opt
        self.model = BayesWrapper(opt)
        for parameter in self.model.classifier.parameters():
            parameter.requires_grad = False
        for parameter in self.model.cls_layer.parameters():
            parameter.requires_grad = False

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizers = [
            SGAdaHMC(
                mlp.parameters(),
                config={
                    "lr": opt.lr,
                    "alpha": 0,
                    "gamma": 0.01,
                    "L": 30,
                    "T": 1e-5,
                    "tao": 2,
                    "C": 1,
                },
            )
            for mlp in self.model.appended_model_list
        ]
        self.output_dir = opt.appmodel_ckpt_root
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_steps = 0

    def _checkpoint_path(self, model_no):
        return os.path.join(
            self.output_dir,
            f"{model_no}{self.opt.appmodel_ckpt_name}",
        )

    def _save_checkpoint(self, model_no):
        path = self._checkpoint_path(model_no)
        torch.save(self.model.appended_model_list[model_no].state_dict(), path)
        print(f"Saved Bayesian checkpoint: {path}")

    def _accuracy(self, loader, model_no):
        self.model.eval(model_No=model_no)
        labels, predictions = [], []
        with torch.no_grad():
            for images, batch_labels, _ in loader:
                output = (
                    self.model.forward_logits(
                        images.to(self.model.device), model_No=model_no
                    )
                    .flatten()
                    .sigmoid()
                )
                predictions.extend((output >= 0.5).cpu().numpy())
                labels.extend(batch_labels.numpy())
        if not labels:
            raise ValueError("Validation dataset is empty.")
        return float(np.mean(np.asarray(labels) == np.asarray(predictions)))

    def train_model(self, train_loader, val_loader, *_):
        best_val_acc = [None] * len(self.optimizers)
        print(f"#training images batches = {len(train_loader)}")

        for epoch in range(self.opt.niter):
            for model_no, optimizer in enumerate(self.optimizers):
                self.model.train(model_No=model_no)
                losses = []
                for images, labels, _ in train_loader:
                    images = images.to(self.model.device)
                    labels = labels.to(self.model.device).float()
                    optimizer.zero_grad(set_to_none=True)
                    output = self.model.forward_logits(images, model_No=model_no)
                    loss = self.loss_fn(output.flatten(), labels)
                    loss.backward()
                    optimizer.step()
                    self.total_steps += 1
                    losses.append(loss.detach().item())

                accuracy = self._accuracy(val_loader, model_no)
                print(
                    f"Epoch {epoch + 1}/{self.opt.niter}, model {model_no}: "
                    f"train_loss={np.mean(losses):.6f}, val_acc={accuracy:.4%}"
                )
                if best_val_acc[model_no] is None or accuracy > best_val_acc[model_no]:
                    best_val_acc[model_no] = accuracy
                    self._save_checkpoint(model_no)
