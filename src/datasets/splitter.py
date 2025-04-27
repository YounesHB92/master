import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from tqdm import tqdm
import shutil

load_dotenv()


class Splitter:
    def __init__(self, config, test_val_ratio, classes=None):
        """
        Initializes the Splitter with image and mask paths and configuration.

        Args:
            image_paths (list): List of file paths to the input images.
            mask_paths (list): List of file paths to the corresponding segmentation masks.
            config (dict): Configuration dictionary containing split ratios and other parameters.
        """
        self.raw_images_path = os.path.join(os.getenv("RAW_DATA_DIR"), "images")
        self.raw_masks_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
        self.config = config
        self.test_val_ratio = test_val_ratio
        self.classes = classes

    def segmentation_spliter(self):
        # checking images and masks path
        self.check_images_files(self.raw_images_path)
        images_files = os.listdir(self.raw_images_path)
        self.check_images_files(self.raw_masks_path)
        masks_files = os.listdir(self.raw_masks_path)

        # checking the split directory if it's full or not
        if len(os.listdir(os.getenv("SPLIT_DATA_DIR"))) > 0:
            raise ValueError("The split directory is not empty. Please clear it before proceeding.")

        x_train, x_test, y_train, y_test = train_test_split(images_files, masks_files, test_size=self.test_val_ratio,
                                                            random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.test_val_ratio,
                                                          random_state=42)
        sets = {
            "train": {
                "images": [file for file in x_train],
                "masks": [file for file in y_train]
            },
            "val": {
                "images": [file for file in x_val],
                "masks": [file for file in y_val]
            },
            "test": {
                "images": [file for file in x_test],
                "masks": [file for file in y_test]
            }
        }

        # Create directories for train, val, and test sets
        for set_name in sets.keys():
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images"))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks"))

        # copy images and masks to the respective directories
        for set_name, set_data in sets.items():
            self.copy_images_masks(set_name, set_data)

    def check_images_files(self, path):
        print("Checking path: ", path)
        files = os.listdir(path)
        for file in tqdm(files, desc="Checking files", total=len(files)):
            if not file.endswith(('.jpg', '.png', '.jpeg')):
                raise ValueError(f"Invalid file format: {file}. Only .jpg, .png, and .jpeg are allowed.")
        print("All files are valid.")

    def copy_images_masks(self, set_name, set_data):
        print("Copying images and masks for set: ", set_name)
        for image_file in tqdm(set_data["images"], desc="Copying images", total=len(set_data["images"])):
            source_path = os.path.join(self.raw_images_path, image_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images", image_file)
            shutil.copyfile(source_path, destination_path)
        for mask_file in tqdm(set_data["masks"], desc="Copying masks", total=len(set_data["masks"])):
            source_path = os.path.join(self.raw_masks_path, mask_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks", mask_file)
            shutil.copyfile(source_path, destination_path)
