import os
import random

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SynImageDataset(Dataset):
    def __init__(
        self,
        real_dir,
        fake_dir,
        opt,
        process_fn,
        max_sample=None,
    ):
        self.opt = opt
        self.process_fn = process_fn
        if max_sample is None:
            max_sample = getattr(opt, "max_sample", None)

        self.reals = self._get_image_paths(real_dir)
        self.fakes = self._get_image_paths(fake_dir)

        if max_sample is not None:
            random.shuffle(self.reals)
            random.shuffle(self.fakes)
            self.reals = self.reals[:max_sample]
            self.fakes = self.fakes[:max_sample]

        self.images = self.reals + self.fakes
        self.labels = [0] * len(self.reals) + [1] * len(self.fakes)

        print(f"Load {len(self.reals)} reals and {len(self.fakes)} fakes.")

    @staticmethod
    def _get_image_paths(root):
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_paths = []
        for directory, _, filenames in os.walk(root):
            for filename in filenames:
                if os.path.splitext(filename)[1].lower() in extensions:
                    image_paths.append(os.path.join(directory, filename))
        return sorted(image_paths)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            label = self.labels[idx]
            image = Image.open(image_path).convert("RGB")

            image = self.process_fn(image, self.opt, label, image_path)
            return image
        except Exception as e:
            print(e)
            return self[idx + 1]


class CleanSampleDataset(Dataset):
    def __init__(self, images: list, labels: list, opt, process_fn):
        super().__init__()
        self.images = images
        self.labels = labels
        self.opt = opt
        self.process_fn = process_fn

        print(
            f"Load {len(self.labels) - self.labels.sum()} reals and {self.labels.sum()} fakes."
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        return self.process_fn(image, self.opt, label, image_path)
