import os
from random import randint

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

from src.datasets import SplitterCore
from src.utils import load_env_variables, print_indented

_ = load_env_variables()


class FeatureLoader(SplitterCore):
    def __init__(self, features_file, test_val_ratio=0.2, force_directory=False, random_state=42, *args, **kwargs):
        super().__init__(test_val_ratio, force_directory, random_state, *args, **kwargs)
        self.features_path = os.getenv("FEATURES_DIR")
        self.features_file = features_file
        self.features_df = self.load_features()
        self.classes_df = self.load_classes()
        self.do_the_checks()

        self.features = None
        self.run()

    def load_features(self):
        features_path = os.path.join(self.features_path, self.features_file)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        features = pd.read_csv(features_path)
        print("\nFeatures loaded from: {}".format(self.features_file))
        print_indented("Number of features: {}".format(len(features.columns)), level=1)
        print_indented("Number of samples: {}".format(len(features)), level=1)
        return features

    def load_classes(self):
        classes_path = os.path.join(os.getenv("RAW_DATA_DIR"), "classes", "crack_types.csv")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        classes = pd.read_csv(classes_path)

        # temporary drop complexity and causes
        classes.drop(["complexity", "causes"], axis=1, inplace=True)

        print("\nClasses loaded from: {}".format(os.path.basename(classes_path)))
        print_indented("Number of columns: {}".format(len(classes.columns)), level=1)
        print_indented("Number of rows: {}".format(len(classes.index)), level=1)
        return classes

    def do_the_checks(self):
        if len(self.features_df) != len(self.classes_df):
            raise ValueError("Number of samples in features and classes do not match.")

    def analyze_one_sample(self, plot=False):
        rand_index = randint(0, len(self.features_df) - 1)
        image_file = self.features_df.loc[rand_index, "image"]
        print("\nAnalyzing sample: {}".format(image_file))

        class_index = self.classes_df[self.classes_df["image"] == image_file].index
        class_ = self.classes_df.loc[class_index, "type"].values[0]
        print_indented("Class: {}".format(class_), level=1)

        # load_mask
        masks_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
        mask = cv.imread(os.path.join(masks_path, image_file), 0)
        box_coords = [
            "bbox-0",
            "bbox-1",
            "bbox-2",
            "bbox-3",
        ]  # [min_row, min_col, max_row, max_col]
        coords = self.features_df.loc[rand_index, box_coords].values
        min_row, min_col, max_row, max_col = coords
        width = max_col - min_col
        height = max_row - min_row

        if plot:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.imshow(mask)
            ax.axis('off')

            rect = patches.Rectangle(
                (min_col, min_row),
                width,
                height,
                linewidth=2,
                edgecolor='r',
                facecolor='none',
            )

            ax.add_patch(rect)
            title = f"mask file: {image_file}, class: {class_}"
            plt.title(title)
            plt.show()

    def find_set(self, x, train_list, val_list, test_list):
        if x in train_list:
            return "train"
        elif x in val_list:
            return "val"
        elif x in test_list:
            return "test"
        else:
            return None

    def run(self):
        if self.sets is None:
            raise Exception("Something went wrong.")

        train_list = self.sets["train"]["masks"]
        val_list = self.sets["val"]["masks"]
        test_list = self.sets["test"]["masks"]

        self.features_df["set"] = self.features_df["image"].apply(
            lambda x: self.find_set(x, train_list, val_list, test_list))
        self.classes_df["set"] = self.features_df["image"].apply(
            lambda x: self.find_set(x, train_list, val_list, test_list))

        x_train = self.features_df.loc[self.features_df["set"] == "train"]
        x_val = self.features_df.loc[self.features_df["set"] == "val"]
        x_test = self.features_df.loc[self.features_df["set"] == "test"]

        y_train = self.classes_df.loc[self.classes_df["set"] == "train"]
        y_val = self.classes_df.loc[self.classes_df["set"] == "val"]
        y_test = self.classes_df.loc[self.classes_df["set"] == "test"]

        self.features = {
            "x_train": x_train,
            "y_train": y_train,
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
        }
        print("\nFeatures are loaded successfully.")
        print_indented("self.features to be used.", level=1)

        return x_train, x_val, x_test, y_train, y_val, y_test
