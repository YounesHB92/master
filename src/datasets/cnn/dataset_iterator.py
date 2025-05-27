import os
from glob import glob

import albumentations as A
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
import numpy as np
from matplotlib import pyplot as plt

from src.utils import load_env_variables
_ = load_env_variables()


class CnnDatasetIterator(torch.utils.data.Dataset):
    """
    Dataset for CNN crack type classification using pre-segmented mask patches.

    Args:
        set_name (str): 'train', 'val', or 'test'
        patch_size (int): Size to which each image is resized
        augment (bool): Whether to apply augmentation
    """
    def __init__(self, set_name, patch_size, augment=True):
        self.set_name = set_name
        self.patch_size = patch_size
        self.augment = augment

        self.class_to_idx = None
        self.images_paths = None
        self.labels = None
        self.classes_list = None

        self.create_paths_and_labels()

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = self.load_image(self.images_paths[idx])  # [H, W, 3]
        label = self.labels[idx]

        if self.augment:
            image = self.transform_augmented(image)
        else:
            image = self.transform_no_augment(image)

        return image, label

    def load_image(self, path):
        # Load grayscale and convert to 3-channel by repeating
        image = Image.open(path).convert('L')
        image = np.array(image)  # (H, W)
        image = np.stack([image] * 3, axis=-1)  # (H, W, 3)
        return image

    def transform_augmented(self, image):
        transform = A.Compose([
            A.Resize(self.patch_size, self.patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        return transform(image=image)["image"]

    def transform_no_augment(self, image):
        transform = A.Compose([
            A.Resize(self.patch_size, self.patch_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        return transform(image=image)["image"]

    def create_paths_and_labels(self):
        base_dir = os.path.join(os.getenv("SPLIT_DATA_DIR"), "cnn_splits", self.set_name)
        class_names = sorted(os.listdir(base_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.classes_list = class_names

        self.images_paths = []
        self.labels = []

        for cls_name in class_names:
            cls_dir = os.path.join(base_dir, cls_name)
            cls_images = glob(os.path.join(cls_dir, "*.png"))
            self.images_paths.extend(cls_images)
            self.labels.extend([self.class_to_idx[cls_name]] * len(cls_images))

    def plot_samples(self, n=5):
        for idx in range(n):
            rand_idx = np.random.randint(len(self.images_paths))
            image, label = self.__getitem__(rand_idx)
            image = image.permute(1, 2, 0).numpy()  # convert back to HWC
            fig, ax = plt.subplots(1, 1)
            ax.imshow(image)
            ax.axis('off')
            ax.set_title("Label: {}".format(list(self.class_to_idx.keys())[label]))
            fig.show()




