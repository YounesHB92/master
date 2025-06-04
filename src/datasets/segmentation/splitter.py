import os
import shutil

import pandas as pd
from .. import SplitterCore
from src.utils import general, image, env_

class SegmentationSplitter(SplitterCore):
    def __init__(self, test_val_ratio, force_dir, classes_list, random_state=42):
        super().__init__(test_val_ratio, force_dir, random_state)
        self.force_directory = force_dir
        self.classes_list = classes_list
        self.classes = None
        self.analyze_classes()
        self.crack_types = self.load_crack_types()  # loading crack types here
        self.run()

    def analyze_classes(self):
        if "background" not in self.classes_list:
            raise ValueError("Background class is missing. Please include 'background' in the classes list.")
        if len(self.classes_list) > 2 and len(
                self.classes_list) != 8:  # because we have defined 7 crack types + 1 background
            if "other" not in self.classes_list:
                raise ValueError(
                    "You are not gonna include all the types but other class is missing. Please include 'other' in the classes list. Otherwise, use all the classes.")
        self.classes = {
            "background": 0
        }
        counter = 1
        for cls in self.classes_list:
            if cls != "background":
                self.classes[cls] = counter
                counter += 1

    def load_crack_types(self):
        raw_path = env.get_raw_path()
        return pd.read_csv(os.path.join(raw_path, "classes", "crack_types.csv"))

    def run(self):
        split_dir = env.get_split_path()
        for set_name in self.set_names:
            set_path = os.path.join(split_dir, set_name)
            path = os.path.join(set_path, "seg_masks")
            self.handle_path(path)
            mask_files = self.sets[set_name]["masks"]
            for mask_file in general.tqdm_print(mask_files, desc=f"Rebuilding masks for set {set_name}", total=len(mask_files)):
                rebuilt_mask = image.rebuild_mask(
                    mask_file=mask_file,
                    classes_list=self.classes_list,
                    classes_dict=self.classes,
                    crack_types=self.crack_types
                )
                saving_path = os.path.join(set_path, "seg_masks", mask_file)
                image.save_rebuilt_mask(rebuilt_mask, saving_path)

    def handle_path(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

