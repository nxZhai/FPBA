import os
import time
import torch
import numpy as np
from tqdm import tqdm

from models import get_model
from bayes_wrapper import BayesWrapper
from dataset.process import processing, tfatk_processing
from dataset.dataset import SynImageDataset
from args import BaseArgs
from utils import (
    set_random_seed,
    standard_confusion_matrix, 
    get_accuracy, 
    get_classification_scores
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def eval(opt: BaseArgs, model, dataloader, exp_time):
    start_time = time.time()

    # with torch.no_grad():
    with torch.no_grad():
        y_true, y_pred = [], []
        for index, (images, labels, img_path) in enumerate(tqdm(dataloader, leave=False)):
            images, labels = images.cuda(), labels.cuda()
            
            if opt.model == 'LGrad' and opt.whole:
                images.requires_grad = True

            # pred
            output = model(images).sigmoid().flatten()
            gt = labels.flatten()

            output[output>0.5] = 1
            output[output<0.5] = 0

            y_pred.extend(output.tolist())
            y_true.extend(gt.tolist())

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
    }

    if opt.mode == 'tf_atk':
        output_dict.update({
            "surrogate": opt.surrogate,
            "attack_method": opt.tf_attack,
            "ASR": f"{(1-accuracy):>.4%}",
            "REAL_ASR": f"{(1-real_acc):>.4%}",
            "FAKE_ASR": f"{(1-fake_acc):>.4%}"
        })

    print(
        f"ASR: {(1-accuracy)}\n",
        f"REAL ASR: {(1-real_acc):>.4%}\n",
        f"FAKE ASR: {(1-fake_acc):>.4%}\n"
    )

    return accuracy, precision # for val

def main():

    opt = BaseArgs(explicit_bool=True).parse_args().update_args()
    set_random_seed(seed=opt.seed)
    os.makedirs(opt.results_dir, exist_ok=True)
    exp_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    # init model
    if not opt.bayes:
        model = get_model(opt)
        model.load_weights(opt)
        model = torch.nn.DataParallel(model)
        model.eval().cuda()
    elif opt.bayes:
        model = BayesWrapper(opt=opt)

    # load data
    eval_set = SynImageDataset(
        real_dir=os.path.join(opt.data_root, "0_real"),
        fake_dir=os.path.join(opt.data_root, "1_fake"),
        process_fn=processing if opt.surrogate == "None" else tfatk_processing,
        opt=opt
    )

    data_loader = torch.utils.data.DataLoader(eval_set,
                                            batch_size=opt.batch_size,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=opt.num_workers)

    eval(opt, model, data_loader, exp_time)


if __name__ == "__main__":
    main()