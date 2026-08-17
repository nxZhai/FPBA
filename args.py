import os
import sys
from typing import Literal

from tap import Tap


class BaseArgs(Tap):
    seed: int = 42
    exp_name: str = None
    mode: Literal["train", "test", "attack", "tf_atk"] = None
    model: str = "CNNSpot"

    dataset: str = None
    data_root: str = None
    train_split: str = None
    val_split: str = None

    ckpt: str = None
    results_dir: str = "./results"

    batch_size: int = 64
    num_workers: int = 14
    load_size: int = 256
    crop_size: int = 224
    no_crop: bool = False
    no_resize: bool = False
    no_flip: bool = False

    init_type: str = (
        "normal"
    )
    init_gain: float = 0.2

    dcta_ckpt_dir: str = "./checkpoints"
    dct_mean: str = None
    dct_var: str = None
    patch_num: int = 3
    downsampling_prob: float = 0.1

    data_aug: bool = False
    rz_interp: str = "bilinear"
    blur_prob: float = 0.1
    blur_sig: str = "0.0, 3.0"
    jpg_prob: float = 0.1
    jpg_method: str = "cv2,pil"
    jpg_qual: str = "30, 100"
    earlystop: bool = False
    bayes: bool = False
    csgld: bool = False
    bayes_model_num: int = 3
    appmodel_ckpt_root: str = None
    appmodel_ckpt_name: str = None

    niter: int = 3
    lr: float = 1e-4
    beta1: float = 0.9
    optim: str = "adam"
    earlystop_epoch: int = 5
    save_epoch_freq: int = 5
    save_latest_freq: int = 2000
    continue_train: bool = False
    loss_freq: int = 100
    max_sample: int = None
    checkpoints_dir: str = "./checkpoints"
    whole: bool = False
    no_grad: bool = (
        False
    )

    tf_attack: str = None
    surrogate: str = None

    def update_args(self):
        if self.model == "DCTA":
            if not self.dcta_ckpt_dir:
                raise ValueError("DCTA requires --dcta_ckpt_dir.")
            self.dct_mean = os.path.join(self.dcta_ckpt_dir, f"mean_{self.dataset}.pt")
            self.dct_var = os.path.join(self.dcta_ckpt_dir, f"var_{self.dataset}.pt")
            for path in (self.dct_mean, self.dct_var):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"DCTA statistic file not found: {path}")
        else:
            self.dct_mean = self.dct_var = None
        if self.train_split is None:
            self.train_split = f"{self.dataset}_train"
        if self.val_split is None:
            self.val_split = f"{self.dataset}_val"

        if self.model in ["LGrad", "LNP"]:
            self.blur_prob = self.jpg_prob = 0.0

        self.rz_interp = self.rz_interp.split(",")
        self.blur_sig = [float(s) for s in self.blur_sig.split(",")]
        self.jpg_method = self.jpg_method.split(",")
        self.jpg_qual = [int(s) for s in self.jpg_qual.split(",")]
        if len(self.jpg_qual) == 2:
            self.jpg_qual = list(range(self.jpg_qual[0], self.jpg_qual[1] + 1))
        elif len(self.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.results_dir = os.path.join(
            self.results_dir,
            self.dataset,
            self.surrogate if self.surrogate else self.model,
            self.mode,
            self.exp_name,
        )

        return self


class AttackArgs(BaseArgs):
    attack: str = None
    adv_data_path: str = None

    epsilon: int = 8
    alpha: int = 2
    steps: int = 10
    targeted: bool = False
    decay: float = 1.0
    image_width: int = 224
    momentum: float = 1.0
    rho: float = 0.5
    N: int = 5

    def update_args(self):
        self = super().update_args()
        self.epsilon = self.epsilon / 255.0
        self.alpha = self.alpha / 255.0

        assert self.epsilon >= 0.0 and self.epsilon <= 1.0
        assert self.alpha >= 0.0 and self.alpha <= 1.0

        self.results_dir = self.adv_data_path

        if os.path.exists(self.adv_data_path):
            print(f"[ERROR] Dir {self.results_dir} existed, shutting down the script.")
            sys.exit(0)

        return self
