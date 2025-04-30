import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from src.utils import load_env_variables
from src.utils import print_indented
import src.utils as utils
import shutil
import pandas as pd

_ = load_env_variables()

class Splitter:
    def __init__(self, task, test_val_ratio, classes_list, force_directory=True):
        self.raw_images_path = os.path.join(os.getenv("RAW_DATA_DIR"), "images")
        self.raw_masks_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
        self.test_val_ratio = test_val_ratio
        self.force_directory = force_directory
        self.classes_list = classes_list
        self.classes = None
        self.analyze_classes()
        self.crack_types = pd.read_csv(
            os.path.join(os.getenv("RAW_DATA_DIR"), "classes", "crack_types.csv"))  # loading crack types here
        self.task = task.lower()
        if self.task == "segmentation":
            self.segmentation_splitter()

    def analyze_classes(self):
        if "background" not in self.classes_list:
            raise ValueError("Background class is missing. Please include 'background' in the classes list.")
        if len(self.classes_list) > 2 & len(self.classes_list) != 8: # because we have defined 7 crack types + 1 background
            if "other" not in self.classes_list:
                raise ValueError("You are not gonna include all the types but other class is missing. Please include 'other' in the classes list. Otherwise, use all the classes.")
        self.classes = {
            "background": 0
        }
        counter = 1
        for cls in self.classes_list:
            if cls != "background":
                self.classes[cls] = counter
                counter += 1

    def segmentation_splitter(self):
        # checking images and masks path
        self.check_images_files(self.raw_images_path)
        images_files = os.listdir(self.raw_images_path)
        self.check_images_files(self.raw_masks_path)
        masks_files = os.listdir(self.raw_masks_path)

        x_train, x_test, y_train, y_test = train_test_split(images_files, masks_files,
                                                            test_size=self.test_val_ratio,
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

        # checking the split directory if it's full or not
        split_dir_files = os.listdir(os.getenv("SPLIT_DATA_DIR"))
        if len(split_dir_files) == 0:
            print("Split directory is empty -> creating new directories.")
            self.create_set_dirs(sets)
            self.copy_sets(sets)

        elif len(split_dir_files) > 0:
            print("\n")
            print("Split directory is not empty.")
            if self.force_directory:
                print("Overwriting existing split directory.")
                print("self.force_directory -> ", self.force_directory)
                for dir_ in split_dir_files:
                    if os.path.isdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), dir_)):
                        shutil.rmtree(os.path.join(os.getenv("SPLIT_DATA_DIR"), dir_))
                    else:
                        os.remove(os.path.join(os.getenv("SPLIT_DATA_DIR"), dir_))

                self.create_set_dirs(sets)
                self.copy_sets(sets)
            else:
                print("Skipping split directory creation and transfer.")

    def check_images_files(self, path):
        print("\nChecking path: ", path)
        files = os.listdir(path)
        for file in utils.tqdm_print(files, desc="Checking files", total=len(files)):
            if not file.endswith(('.jpg', '.png', '.jpeg')):
                raise ValueError(f"Invalid file format: {file}. Only .jpg, .png, and .jpeg are allowed.")
        print("All files are valid.")

    def copy_images_masks(self, set_name, set_data):
        print("Copying images and masks for set: ", set_name)
        for image_file in utils.tqdm_print(set_data["images"], desc="Copying images", total=len(set_data["images"])):
            source_path = os.path.join(self.raw_images_path, image_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images", image_file)
            shutil.copyfile(source_path, destination_path)
        for mask_file in utils.tqdm_print(set_data["masks"], desc="Copying masks", total=len(set_data["masks"])):
            rebuilt_mask = self.rebuild_mask(mask_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks", mask_file)
            Image.fromarray(rebuilt_mask).save(destination_path)

    def create_set_dirs(self, sets):
        for set_name in sets.keys():
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images"))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks"))

    def copy_sets(self, sets):
        for set_name, set_data in sets.items():
            self.copy_images_masks(set_name, set_data)

    def rebuild_mask(self, mask_file):
        # load_mask
        mask_path = os.path.join(self.raw_masks_path, mask_file)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        rebuilt_mask = np.zeros_like(mask, dtype=np.uint8)
        if len(self.classes_list) == 2:
            rebuilt_mask[mask!=0] = 1
            return rebuilt_mask.astype("uint8")
        else:
            index_ = self.crack_types[self.crack_types["Image"] == mask_file].index
            crack_type = self.crack_types.loc[index_, "Type"].values[0]
            if crack_type not in self.classes.keys():
                crack_type = "other"
            rebuilt_mask[mask != 0] = self.classes[crack_type]
            return rebuilt_mask.astype("uint8")



