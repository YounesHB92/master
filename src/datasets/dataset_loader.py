import cv2 as cv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetLoader(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing images and masks for semantic segmentation.

    Attributes:
        image_paths (list): List of file paths to the input images.
        mask_paths (list): List of file paths to the corresponding segmentation masks.
        path_size (int): The size to which images and masks will be resized.
        augment (bool): Whether to apply data augmentation to the images and masks.
        mode (str): The task mode, either 'binary' for binary segmentation or 'multiclass' for multi-class segmentation.
    """

    def __init__(self, image_paths, mask_paths, path_size, augment=True, mode='multiclass'):
        """
        Initializes the DatasetLoader.

        Args:
            image_paths (list): List of file paths to the input images.
            mask_paths (list): List of file paths to the corresponding segmentation masks.
            path_size (int): The size to which images and masks will be resized.
            augment (bool, optional): Whether to apply data augmentation. Defaults to True.
            mode (str, optional): The task mode ('binary' or 'multiclass'). Defaults to 'multiclass'.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.path_size = path_size
        self.augment = augment
        self.mode = mode.lower()  # force lower case

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the specified index, applies transformations, and returns them.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the transformed image (torch.Tensor) and mask (torch.Tensor).
        """
        image = self.load_image(self.image_paths[idx])
        mask = self.load_mask(self.mask_paths[idx])

        if self.augment:
            augmented = self.transform_augmented(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            no_augment = self.transform_no_augment(image=image, mask=mask)
            image = no_augment['image']
            mask = no_augment['mask']

        # Set correct mask dtype depending on task
        if self.mode == 'binary':
            mask = mask.float()  # For BCEWithLogitsLoss / BinaryFocal
        elif self.mode == 'multiclass':
            mask = mask.long()  # For CrossEntropyLoss / FocalLoss (multi-class)
        else:
            raise ValueError(f"Unknown mode {self.mode}. Use 'binary' or 'multiclass'.")

        return image, mask

    def load_image(self, path):
        """
        Loads an image from the specified file path and converts it to RGB format.

        Args:
            path (str): The file path to the image.

        Returns:
            numpy.ndarray: The loaded image in RGB format.
        """
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return image

    def load_mask(self, path):
        """
        Loads a segmentation mask from the specified file path in grayscale format.

        Args:
            path (str): The file path to the mask.

        Returns:
            numpy.ndarray: The loaded mask in grayscale format.
        """
        mask = cv.imread(path, 0)  # grayscale
        return mask

    def transform_augmented(self, image, mask):
        """
        Applies data augmentation transformations to the image and mask.

        Args:
            image (numpy.ndarray): The input image.
            mask (numpy.ndarray): The corresponding segmentation mask.

        Returns:
            dict: A dictionary containing the augmented image and mask.
        """
        transform = A.Compose([
            A.Resize(self.path_size, self.path_size),
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
                var_limit=(10.0, 50.0), p=0.2
            ),
            A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50, p=0.2
            ),
            A.GridDistortion(p=0.2),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)

    def transform_no_augment(self, image, mask):
        """
        Applies basic transformations (resizing and normalization) to the image and mask.

        Args:
            image (numpy.ndarray): The input image.
            mask (numpy.ndarray): The corresponding segmentation mask.

        Returns:
            dict: A dictionary containing the transformed image and mask.
        """
        transform = A.Compose([
            A.Resize(self.path_size, self.path_size),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),
            ToTensorV2()
        ])
        return transform(image=image, mask=mask)
