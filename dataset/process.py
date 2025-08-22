from io import BytesIO
from random import random, choice

import cv2
from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


def processing(img, opt, label, img_path):
    if opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    if opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.crop_size)

    trans = transforms.Compose([
                rz_func,
                crop_func,
                transforms.ToTensor(),
                ])

    return img, label, img_path


rz_dict = {'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS,
            'nearest': Image.NEAREST}


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def custom_resize(img, opt):

    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.load_size, opt.load_size), interpolation=rz_dict[interp])

def func_process(img, opt):
    # Resize
    if opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    # Crop
    if opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.crop_size)

    trans = transforms.Compose([
                rz_func,
                crop_func,
                transforms.ToTensor(),
                ])

    return trans(img)


def tfatk_processing(img, opt, label, img_path):
    """
    Use this func to train and test models according to their papers,
    """
    # spatial-based models

    # resize adv samples to 224x224.
    opt.no_resize=False
    opt.load_size=224
    # do not perform CenterCrop operation.
    opt.no_crop = True

    img = func_process(img, opt)
    return img, label, img_path
