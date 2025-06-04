import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import print_indented
from src.utils import image


class SegmentationDatasetLoader:
    def __init__(self, dataset, classes, batch_size, shuffle=True, num_workers=4, report=True, show_samples=False):
        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.report = report
        self.show_samples = show_samples
        self.loader = self.create_loader()
        if self.report:
            self.print_report()
            self.print_random_sample()

    def print_report(self):
        print("\n")
        print("-" * 50)
        print(f"\033[1m{self.dataset.set_name.upper()}\033[0m dataset Loader Report:")
        print_indented(f"Number of samples: {len(self.dataset.images_paths)}")
        print_indented(f"Batch size: {self.batch_size}")
        print_indented(f"Shuffle: {self.shuffle}")
        print_indented(f"Number of workers: {self.num_workers}")
        print("-" * 50)

    def create_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def print_random_sample(self):
        print("Random samples from dataset:")
        idx = torch.randint(0, len(self.dataset.images_paths), (1,)).item()
        sample = self.dataset[idx]

        # handle different types of datasets
        if isinstance(sample, tuple) and len(sample) >= 2:
            image, mask = sample[0], sample[1]
            idxes = np.unique(mask[mask != 0])
            crack_types = []
            for idx in idxes:
                crack_types.append(list(self.classes.keys())[idx])
            print_indented(f"Crack types in the sample: {', '.join(crack_types)}")
            print_indented(f"Image shape: {image.shape}")
            print_indented(f"Mask shape: {mask.shape}")

        if hasattr(self.dataset, 'images_paths') and hasattr(self.dataset, 'masks_paths'):
            image_file = os.path.basename(self.dataset.images_paths[idx])
            mask_file = os.path.basename(self.dataset.masks_paths[idx])
            print_indented(f"Image file: {image_file}")
            print_indented(f"Mask file: {mask_file}")
            if self.show_samples:
                self.show_samples_()
        print("-" * 50)

    import matplotlib.pyplot as plt

    def show_samples_(self):
        for i in range(self.show_samples):
            idx = torch.randint(0, len(self.dataset.images_paths), (1,)).item()
            image_path = self.dataset.images_paths[idx]
            mask_path = self.dataset.masks_paths[idx]

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_colorized = image.visualize_mask(mask)

            if mask_colorized.shape[:2] != img.shape[:2]:
                mask_colorized = cv2.resize(mask_colorized, (img.shape[1], img.shape[0]))

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(img)
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(mask_colorized)
            axs[1].set_title("Mask")
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()



