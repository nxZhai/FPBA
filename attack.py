import os
import sys
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from tap import Tap

# from options.attack_options import AttackOptions
from args import AttackArgs

from fpba import FPBA
from models import get_model
from bayes_wrapper import BayesWrapper
from dataset.process import processing
from dataset.dataset import SynImageDataset, CleanSampleDataset
from utils import (
    set_random_seed,
    standard_confusion_matrix, 
    get_accuracy, 
    get_classification_scores
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def save_images(new_root, adv_tens, gt_tens, paths):

    adv_tens = (adv_tens.detach().cpu() * 255.).clamp(0., 255.)
    adversaries = adv_tens.permute((0,2,3,1)).numpy().astype(np.uint8)
    # adversaries = (adv_tens.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)

    for idx in tqdm(range(len(paths)), leave=False):
        
        if 'gan' in paths[idx].split('/'):
            paths_split = paths[idx].split('/')
            assert paths_split[-5] == 'gan' or paths_split[-4] == 'gan' # for test or exp subset
            # paths_split[-3] is subclass name for wang2020 ProGAN testset.
            filename = f"{paths_split[-3]}_{paths_split[-1]}"
        else:
            # JPEG is a type of lossy compression for GenImage REAL Images
            filename = os.path.basename(paths[idx])
            if "JPEG" in filename:
                filename = filename.replace(".JPEG", ".png")

        gt = "0_real" if not gt_tens[idx] else "1_fake"
        save_path = os.path.join(new_root, gt)
        os.makedirs(save_path, exist_ok=True)

        Image.fromarray(adversaries[idx]).save(os.path.join(save_path, filename))


def select_corr_clndata(opt: AttackArgs, model, dataloader):
    corr_classify_count = 0
    results = np.empty(len(dataloader.dataset))
    # LGrad for white test need computing grad as one end2end model
    opt.no_grad = False if (opt.model == "LGrad" and opt.whole) else True
    with torch.no_grad():
        for index, (images, labels, img_path) in enumerate(tqdm(dataloader, leave=False)):
            images, labels = images.cuda(), labels.cuda()
            # pred
            output = model(images).sigmoid().flatten()
            gt = labels.flatten()
            output[output>0.5] = 1
            output[output<0.5] = 0
            
            diff = (output==gt)
            corr_classify_count = corr_classify_count + diff.sum()
            results[index*(labels.size(0)):(index+1)*(labels.size(0))] = diff.cpu().numpy()

    images_set = np.array(list(dataloader.dataset.images))
    labels_set = np.array(list(dataloader.dataset.labels))
    corr_images = images_set[results.astype(bool)]
    corr_labels = labels_set[results.astype(bool)]

    corrdata_set = CleanSampleDataset(
        images=corr_images, 
        labels=corr_labels, 
        opt=opt, 
        process_fn=processing
    )

    corrdata_loader = torch.utils.data.DataLoader(
        corrdata_set,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers
    )

    return corrdata_loader


def attack(opt: AttackArgs, model, dataloader, exp_time):
    start_time = time.time()
    attacker = FPBA(opt, model)

    try:
        os.makedirs(opt.adv_data_path)
    except FileExistsError as e:
        print(f"dir {opt.adv_data_path} existed.")
        sys.exit(1)

    y_true, y_pred = [], []
    for index, (images, labels, img_path) in enumerate(tqdm(dataloader, leave=False)):
        images, labels = images.cuda(), labels.cuda()

        # attack
        adv_images = attacker.attack(images, labels)

        save_images(opt.adv_data_path, adv_images, labels, img_path)

        # pred
        with torch.no_grad():
            output = model(adv_images).sigmoid().flatten()
            gt = labels.flatten()

            output[output>0.5] = 1
            output[output<0.5] = 0

            y_pred.extend(output.tolist())
            y_true.extend(gt.tolist())

    print(f"[White Attack]: From {opt.data_root} to {opt.adv_data_path}.")
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    [[tp, fp], [fn, tn]] = standard_confusion_matrix(y_true, y_pred)
    accuracy, correct_count = get_accuracy(y_true, y_pred)
    fake_acc, real_acc, precision, recall, f1_score = get_classification_scores(y_true, y_pred)

    end_time = time.time()

    output_dict = {
        'mode': opt.mode,   
        'exp_name': opt.exp_name,
        'time': exp_time,
        'dataset': opt.data_root.split("/")[-1], # opt.dataset,
        'model': opt.model,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'accuracy':     f"{accuracy:.4%}",
        'precision':    f"{precision:.4%}",
        'recall':       f"{recall:.4%}",
        'f1_score':     f"{f1_score:.4%}",
        'real_acc':     f"{real_acc:.4%}",
        'fake_acc':     f"{fake_acc:.4%}",
        'attacker':     opt.attack,
        'targeted':     opt.targeted,
        'ASR':          f"{(1-accuracy):>10.4%}",
        'REAL_ASR':     f"{(1-real_acc):>10.4%}",
        'FAKE_ASR':     f"{(1-fake_acc):>10.4%}"
    }

    print(
        f'ASR: {(1-accuracy):>10.4%}\n',
        f'REAL_ASR {(1-real_acc):>10.4%}\n',
        f'FAKE_ASR {(1-fake_acc):>10.4%}\n'
    )



def main():

    opt = AttackArgs(explicit_bool=True).parse_args().update_args()
    set_random_seed(seed=opt.seed)

    exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    # 1. model
    if not opt.bayes:
        model = get_model(opt)
        model.load_weights(opt)
        if opt.attack != "SSAH":
            model = torch.nn.DataParallel(model)
        model.eval().cuda()
    elif opt.bayes:
        model = BayesWrapper(opt=opt)
    # collect images which are classified by surrogate firstly.
    # if opt.attack == 'StealthDiff':
    #     opt.compute_real = False
    #     opt.compute_fake = True

    # 2. dataset
    eval_set = SynImageDataset(
        real_dir=os.path.join(opt.data_root, "0_real"),
        fake_dir=os.path.join(opt.data_root, "1_fake"),
        process_fn=processing,
        opt=opt
    )

    data_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=opt.num_workers
    )

    # 3. 选择识别正确的样本
    corrdata_loader = select_corr_clndata(opt, model, data_loader)
    # 4. attack!
    attack(opt, model, corrdata_loader, exp_time)


if __name__ == "__main__":
    main()