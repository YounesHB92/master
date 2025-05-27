import os

import torch

from src.utils import print_indented


class CnnDatasetLoader:
    """
    CNN Dataset Loader
    Args:
        dataset: the dataset iterator object
        batch_size: the batch size
        shuffle: whether to shuffle the dataset or not
        num_workers: the number of workers
        report: if you would like to double-check the loadings
    Attributes:
        loader: is the dataloader object to path to trainer
    """
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=4, report=True):
        self.dataset = dataset
        self.classes = self.dataset.class_to_idx
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

        image, label = sample[0], sample[1]
        print_indented(f"Chosen sample type: {list(self.classes.keys())[label]}")
        print_indented(f"Image file: {self.dataset.images_paths[idx]}")
        print_indented(f"Mask shape: {image.shape}")
        print("-" * 50)
