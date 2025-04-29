import os

import albumentations as A
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from src.utils import load_env_variables
import numpy as np
_ = load_env_variables()


class DatasetIterator(torch.utils.data.Dataset):
    def __init__(self, set_name, patch_size, augment=True):
        self.set_name = set_name
        self.create_paths()
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.images_paths[idx])
        mask = self.load_mask(self.masks_paths[idx])

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image shape {image.shape[:2]} and mask shape {mask.shape[:2]} do not match!\nImage: {self.images_paths[idx]}\nMask: {self.masks_paths[idx]}"
            )


        if self.augment:
            augmented = self.transform_augmented(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            no_augment = self.transform_no_augment(image=image, mask=mask)
            image = no_augment['image']
            mask = no_augment['mask']

        mask = mask.long()

        return image, mask

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def load_mask(self, path):
        mask = Image.open(path).convert('L')
        return np.array(mask)

    def transform_augmented(self, image, mask):
        transform = A.Compose([
            A.Resize(self.patch_size, self.patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
                p=0.5, border_mode=0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(80, 120), p=0.3
            ),
            A.GaussNoise(
                p=0.2
            ),
            A.ElasticTransform(
                alpha=1, sigma=50, p=0.2
            ),
            A.GridDistortion(p=0.2),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)

    def transform_no_augment(self, image, mask):
        transform = A.Compose([
            A.Resize(self.patch_size, self.patch_size),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)

    def create_paths(self):
        images_dir = os.path.join(os.getenv("SPLIT_DATA_DIR"), self.set_name, "images")
        masks_dir = os.path.join(os.getenv("SPLIT_DATA_DIR"), self.set_name, "masks")

        images_files = sorted(os.listdir(images_dir))
        masks_files = sorted(os.listdir(masks_dir))

        self.images_paths = [
            os.path.join(images_dir, file) for file in images_files
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.masks_paths = [
            os.path.join(masks_dir, file) for file in masks_files
            if file.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if len(self.images_paths) != len(self.masks_paths):
            raise ValueError(
                f"Number of images ({len(self.images_paths)}) and masks ({len(self.masks_paths)}) do not match!")

