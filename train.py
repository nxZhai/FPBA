import copy
import os

import torch

from args import BaseArgs
from dataset.dataset import SynImageDataset
from dataset.process import processing
from trainers.bayes_trainer import BayesTrainer
from trainers.earlystop_trainer import EarlystopTrainer
from utils import set_random_seed


def main():
    opt = BaseArgs(explicit_bool=True).parse_args().update_args()
    if opt.mode != "train":
        raise ValueError("train.py requires --mode train.")
    if opt.model not in {"CNNSpot", "MobileNet"}:
        raise ValueError("train.py supports CNNSpot and MobileNet only.")
    if opt.earlystop == opt.bayes:
        raise ValueError("Choose exactly one of --earlystop or --bayes.")

    set_random_seed(opt.seed)
    val_opt = copy.deepcopy(opt)
    val_opt.mode = "val"

    train_set = SynImageDataset(
        os.path.join(opt.data_root, opt.train_split, "0_real"),
        os.path.join(opt.data_root, opt.train_split, "1_fake"),
        opt,
        processing,
    )
    val_set = SynImageDataset(
        os.path.join(opt.data_root, opt.val_split, "0_real"),
        os.path.join(opt.data_root, opt.val_split, "1_fake"),
        val_opt,
        processing,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=opt.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers,
    )

    if opt.earlystop:
        trainer = EarlystopTrainer(opt, val_opt)
    else:
        trainer = BayesTrainer(opt, val_opt)
    trainer.train_model(train_loader, val_loader)


if __name__ == "__main__":
    main()
