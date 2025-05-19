import os

import torch

from src.utils import print_indented


class DatasetLoader:
    def __init__(self, dataset, classes, batch_size, shuffle=True, num_workers=4, report=True):
        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.report = report
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
            idx = mask[mask!=0].reshape(-1).tolist()[0]
            print_indented(f"Chosen sample type: {list(self.classes.keys())[idx]}")
            print_indented(f"Image shape: {image.shape}")
            print_indented(f"Mask shape: {mask.shape}")

        if hasattr(self.dataset, 'images_paths') and hasattr(self.dataset, 'masks_paths'):
            image_file = os.path.basename(self.dataset.images_paths[idx])
            mask_file = os.path.basename(self.dataset.masks_paths[idx])
            print_indented(f"Image file: {image_file}")
            print_indented(f"Mask file: {mask_file}")
        print("-" * 50)
