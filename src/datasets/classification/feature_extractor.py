import os

import pandas as pd
from skimage.measure import regionprops_table
from tqdm import tqdm

from src.utils import load_env_variables
from src.datasets.feature_core import FeatureCore

_ = load_env_variables()

class FeatureExtractor(FeatureCore):
    def __init__(self, print_, save=True, saving_name=None):
        super().__init__(print_)
        self.save = save
        self.saving_name = saving_name
        self.features = None

    def extract_features(self, mask_path):
        mask = self._load_mask(mask_path)
        region_table = regionprops_table(mask, properties=self.features_list)
        return region_table

    def run_single(self):
        region_table = self.extract_features(self.rnd_mask_pth)
        return region_table

    def run_all(self, stopper=None):
        print("Preparing to extract features from all masks:\n\t{}".format(self.masks_path))
        self.features = pd.DataFrame(columns=["image"] + list(self.feature_columns))
        loop = tqdm(self.mask_files, desc="Extracting features from masks", unit=" mask")
        counter = 0
        for mask_file in loop:
            mask_path = os.path.join(self.masks_path, mask_file)
            region_table = self.extract_features(mask_path)
            region_table["image"] = mask_file
            self.features.loc[len(self.features)] = region_table
            counter += 1
            if stopper is not None and counter == stopper:
                break
        if self.save:
            self.save_features()

    def save_features(self):
        features_path = os.getenv("FEATURES_DIR")
        if not os.path.exists(features_path):
            os.makedirs(features_path)
        if self.save and self.saving_name is None:
            raise Exception("Saving name is None, please provide a name.")
        self.features.to_csv(os.path.join(features_path, "{}.csv".format(self.saving_name)), index=False)
        print("Feature saved to: {}".format(os.path.join(features_path, "{}.csv".format(self.saving_name))))
