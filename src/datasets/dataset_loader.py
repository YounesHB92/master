import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class DatasetLoader:
    """
    A class to load and preprocess datasets for training and validation.

    Attributes:
        train_path (str): Path to the training dataset.
        val_path (str): Path to the validation dataset.
        batch_size (int): Number of samples per batch.
        patch_size (int): Size to which images are resized (default is 224).
        augment (bool): Whether to apply experiments augmentation (default is True).
        train_transformer (torchvision.transforms.Compose): Transformations for the training dataset.
        val_transformer (torchvision.transforms.Compose): Transformations for the validation dataset.
        train_dataset (torchvision.datasets.ImageFolder): Training dataset.
        val_dataset (torchvision.datasets.ImageFolder): Validation dataset.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_classes (int): Number of classes in the training dataset.
    """

    def __init__(
            self,
            train_path,
            val_path,
            batch_size,
            patch_size=224,
            augment=True
    ):
        """
        Initializes the DatasetLoader with paths, batch size, patch size, and augmentation settings.

        Args:
            train_path (str): Path to the training dataset.
            val_path (str): Path to the validation dataset.
            batch_size (int): Number of samples per batch.
            patch_size (int, optional): Size to which images are resized. Defaults to 224.
            augment (bool, optional): Whether to apply experiments augmentation. Defaults to True.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.create_transformer()
        self.get_paths_ready()
        self.create_data_loaders()
        self.report()

    def create_transformer(self):
        """
        Creates experiments transformations for training and validation datasets.
        Applies augmentation transformations if `self.augment` is True.
        """
        base_transforms = [
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        if self.augment:
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(self.patch_size, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            ]
            self.train_transformer = transforms.Compose(augment_transforms + base_transforms)
        else:
            self.train_transformer = transforms.Compose(base_transforms)

        self.val_transformer = transforms.Compose(base_transforms)

    def get_paths_ready(self):
        """
        Prepares the training and validation datasets using the specified transformations.
        """
        self.train_dataset = ImageFolder(self.train_path, transform=self.train_transformer)
        self.val_dataset = ImageFolder(self.val_path, transform=self.val_transformer)

    def create_data_loaders(self):
        """
        Creates DataLoaders for the training and validation datasets.
        """
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def report(self):
        """
        Prints a summary of the datasets, including the number of classes and their names.
        """
        self.num_classes = len(self.train_dataset.classes)
        print(f"Datasets are ready!\n\tDetected {self.num_classes} classes: \n\t\t{'\n\t\t'.join(self.train_dataset.classes)}".encode('utf-8').decode('utf-8'))
        print("datasets_.train_loader and datasets_.val_loader to be used!")