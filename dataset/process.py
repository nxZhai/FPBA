import torchvision.transforms as transforms

from dataset.data_augment import custom_resize, data_augment


def processing(img, opt, label, img_path):
    if opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    if opt.mode == "train":
        crop_func = transforms.RandomCrop(opt.crop_size)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.crop_size)

    if opt.mode == "train" and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    trans = transforms.Compose(
        [
            rz_func,
            transforms.Lambda(
                lambda img: (
                    data_augment(img, opt) if opt.mode in ["train", "val"] else img
                )
            ),
            crop_func,
            flip_func,
            transforms.ToTensor(),
        ]
    )

    return trans(img), label, img_path


def func_process(img, opt):
    if opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    if opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.crop_size)

    trans = transforms.Compose(
        [
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ]
    )

    return trans(img)


def tfatk_processing(img, opt, label, img_path):
    """
    Use this func to train and test models according to their papers,
    """

    opt.no_resize = False
    opt.load_size = 224
    opt.no_crop = True

    img = func_process(img, opt)
    return img, label, img_path
