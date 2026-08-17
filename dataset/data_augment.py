from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import gaussian_filter

rz_dict = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def sample_continuous(values):
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return random() * (values[1] - values[0]) + values[0]
    raise ValueError("Expected one or two values.")


def sample_discrete(values):
    if len(values) == 1:
        return values[0]
    return choice(values)


def gaussian_blur(image, sigma):
    for channel in range(image.shape[2]):
        gaussian_filter(image[:, :, channel], output=image[:, :, channel], sigma=sigma)


def cv2_jpg(image, quality):
    image = image[:, :, ::-1]
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, params)
    decoded = cv2.imdecode(encoded, 1)
    return decoded[:, :, ::-1]


def pil_jpg(image, quality):
    output = BytesIO()
    Image.fromarray(image).save(output, format="jpeg", quality=quality)
    result = np.array(Image.open(output))
    output.close()
    return result


_jpeg_methods = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(image, quality, method):
    return _jpeg_methods[method](image, quality)


def data_augment(image, opt):
    image = np.array(image)
    if random() < opt.blur_prob:
        gaussian_blur(image, sample_continuous(opt.blur_sig))
    if random() < opt.jpg_prob:
        image = jpeg_from_key(
            image,
            sample_discrete(opt.jpg_qual),
            sample_discrete(opt.jpg_method),
        )
    return Image.fromarray(image)


def custom_resize(image, opt):
    interpolation = sample_discrete(opt.rz_interp)
    return TF.resize(
        image, (opt.load_size, opt.load_size), interpolation=rz_dict[interpolation]
    )
