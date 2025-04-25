import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class DatasetLoader:
    def __init__(
            self,
            train_path,
            val_path,
            batch_size,
            augment=True
    ):
        self.train_path = train_path
        self.val_path = val_path
        self.augment = augment
        self.bath_size = batch_size
        self.create_transformer()
        self.get_paths_ready()
        self.create_data_loaders()
        self.report()

    def create_transformer(self):
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]

        if self.augment:
            augment_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ]
            self.train_transformer = transforms.Compose(augment_transforms + base_transforms)
        else:
            self.train_transformer = transforms.Compose(base_transforms)

        self.val_transformer = transforms.Compose(base_transforms)

    def get_paths_ready(self):
        self.train_dataset = ImageFolder(self.train_path, transform=self.train_transformer)
        self.val_dataset = ImageFolder(self.val_path, transform=self.val_transformer)

    def create_data_loaders(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.bath_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.bath_size, shuffle=False)

    def report(self):
        self.num_classes = len(self.train_dataset.classes)
        # print(f"Datasets are ready!\n\tDetected {self.num_classes} classes: \n\t\t{'\n\t\t'.join(self.train_dataset.classes)}")
        print("datasets_.train_loader and datasets_.val_loader to be used!")
