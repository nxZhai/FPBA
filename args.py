import os
import sys
from typing import Literal

from tap import Tap


class BaseArgs(Tap):
    seed: int=42 # random seed for models
    exp_name: str=None # experiment name, required
    mode: Literal['train', 'test', 'attack', 'tf_atk']=None
    model: str="CNNSpot"
    
    # paths
    dataset: str=None
    data_root: str=None
    ## train args
    train_split: str=None
    val_split: str=None

    ## output
    ckpt: str=None
    results_dir: str="./results"
    
    # data
    batch_size: int=64
    num_workers: int=14
    load_size: int=256
    crop_size: int=224
    no_crop: bool=False
    no_resize: bool=False
    no_flip: bool=False
    
    # weight init
    init_type: str="normal" # network initialization [normal|xavier|kaiming|orthogonal]
    init_gain: float=0.2 # scaling factor for normal, xavier and orthogonal
    
    # model specifical args
    ## DCTA
    dcta_ckpt_dir: str="/data/zhainx/hades/checkpoints"
    dct_mean: str=None
    dct_var: str=None
    ## PatchCraft
    patch_num: int=3
    downsampling_prob: float=0.1

    # data augmentation
    ## CNNSpot
    data_aug: bool=False
    rz_interp: str="bilinear"
    blur_prob: float=0.1
    blur_sig: str="0.0, 3.0"
    jpg_prob: float=0.1
    jpg_method: str="cv2,pil"
    jpg_qual: str="30, 100"
    ## trainer
    earlystop: bool=False # earlystop training mode
    bayes: bool=False # for train mode, choose bayes training mode, for test/attack/tf_attack, to wrap model with bayes.
    csgld: bool=False # csgmcmc
    ## bayes strategy
    bayes_model_num: int=3
    appmodel_ckpt_root: str=None
    appmodel_ckpt_name: str=None
    ## whole
    whole: bool=False 
    no_grad: bool=False # for 2-stage method LGrad, when treating it as a end2end model.

    # * for transfer attack
    tf_attack: str=None
    surrogate: str=None 
    
    def update_args(self):
        # DCTA's statistical data
        self.dct_mean = os.path.join(self.dcta_ckpt_dir, f"mean_{self.dataset}.pt")
        self.dct_var = os.path.join(self.dcta_ckpt_dir, f"var_{self.dataset}.pt")
        # train
        self.train_split = f"{self.dataset}_train"
        self.val_split = f"{self.dataset}_val"

        if self.model in ["LGrad", "LNP"]:
            self.blur_prob = self.jpg_prob = 0.0

        # CNNSpot's data augment
        self.rz_interp = self.rz_interp.split(',')
        self.blur_sig = [float(s) for s in self.blur_sig.split(',')]
        self.jpg_method = self.jpg_method.split(',')
        self.jpg_qual = [int(s) for s in self.jpg_qual.split(',')]
        if len(self.jpg_qual) == 2:
            self.jpg_qual = list(range(self.jpg_qual[0], self.jpg_qual[1] + 1))
        elif len(self.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        # sanity check
        self.results_dir = os.path.join(
            self.results_dir,
            self.dataset,
            self.surrogate if self.surrogate else self.model,
            self.mode,
            self.exp_name
        )

        return self


class AttackArgs(BaseArgs):
    attack: str=None # attack method
    adv_data_path: str=None
    
    # basic attack settings
    epsilon: int=8 # abs max 8 pixels
    alpha: int=2
    steps: int=10
    targeted: bool=False
    # MIFGSM
    decay: float=1.0
    # SSIFGSM
    image_width: int=224
    momentum: float=1.0
    rho: float=0.5
    N: int=20


    def update_args(self):
        self = super().update_args()
        self.epsilon = self.epsilon/255.
        self.alpha = self.alpha/255.

        assert self.epsilon>=0. and self.epsilon<=1.
        assert self.alpha>=0. and self.alpha<=1.

        self.results_dir = self.adv_data_path

        if os.path.exists(self.adv_data_path):
            print(f'[ERROR] Dir {self.results_dir} existed, shutting down the script.')
            sys.exit(0)

        return self