import os

import numpy as np
import torch
import torch.nn as nn

from models import get_model
from utils.earlystop import EarlyStopping


class EarlystopTrainer:
    def __init__(self, opt, val_opt=None):
        if opt.model not in {"CNNSpot", "MobileNet"}:
            raise ValueError("EarlystopTrainer supports CNNSpot and MobileNet only.")

        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(opt).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        if opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.checkpoint_path = os.path.join(
            opt.checkpoints_dir, f"{opt.model}_{opt.dataset}.pth"
        )
        self.total_steps = 0
        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        if opt.continue_train:
            self._load_checkpoint()
        else:
            self.model.init_fc(init_gain=opt.init_gain)

    def _load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.model.load_state_dict(
            checkpoint.get("model", checkpoint), strict=True
        )
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)

    def _save_checkpoint(self):
        torch.save(
            {
                "model": self.model.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_steps": self.total_steps,
            },
            self.checkpoint_path,
        )
        print(f"Saved checkpoint: {self.checkpoint_path}")

    def _accuracy(self, loader):
        self.model.eval()
        labels, predictions = [], []
        with torch.no_grad():
            for images, batch_labels, _ in loader:
                output = self.model(images.to(self.device)).flatten().sigmoid()
                predictions.extend((output >= 0.5).cpu().numpy())
                labels.extend(batch_labels.numpy())
        if not labels:
            raise ValueError("Validation dataset is empty.")
        return float(np.mean(np.asarray(labels) == np.asarray(predictions)))

    def train_model(self, train_loader, val_loader, *_):
        stopper = EarlyStopping(
            patience=self.opt.earlystop_epoch, delta=-0.001, verbose=True
        )
        for epoch in range(self.opt.niter):
            self.model.train()
            losses = []
            for images, labels, _ in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(images).flatten(), labels)
                loss.backward()
                self.optimizer.step()
                self.total_steps += 1
                losses.append(loss.detach().item())

            accuracy = self._accuracy(val_loader)
            print(
                f"Epoch {epoch + 1}/{self.opt.niter}: "
                f"train_loss={np.mean(losses):.6f}, val_acc={accuracy:.4%}"
            )
            stopper(accuracy, self._save_checkpoint)
            if stopper.early_stop:
                old_lr = self.optimizer.param_groups[0]["lr"]
                if old_lr <= 1e-6:
                    print("Early stopping.")
                    break
                for group in self.optimizer.param_groups:
                    group["lr"] = max(group["lr"] / 10, 1e-6)
                print(f"Learning rate dropped from {old_lr:g}; continuing training.")
                stopper = EarlyStopping(
                    patience=self.opt.earlystop_epoch, delta=-0.001, verbose=True
                )
