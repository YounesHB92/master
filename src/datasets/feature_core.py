import os
import numpy as np
import cv2 as cv
from random import randint
from skimage.measure import regionprops, regionprops_table
from natsort import natsorted


from src.utils import load_env_variables

_ = load_env_variables()

class FeatureCore:
    def __init__(self, print_=True):
        self.print_ = print_
        self.masks_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
        self.mask_files = natsorted(os.listdir(self.masks_path))
        self.rnd_mask_pth = self._random_mask_path()
        self.features_list = self._get_features_list()
        self.feature_columns = self._get_features_columns()

    def _random_mask_path(self):
        mask_path = os.path.join(self.masks_path, self.mask_files[randint(0, len(self.mask_files) - 1)])
        if self.print_:
            print(f"Random mask path: {mask_path}")
        return mask_path

    def _get_features_list(self):
        mask = self._load_mask(self.rnd_mask_pth)
        regions = regionprops(mask)
        region = regions[0]
        features = []
        for prop in region:
            features.append(prop)
        if self.print_:
            print("Feature list provided.")
        return features

    def _load_mask(self, mask_path):
        mask = cv.imread(mask_path, 0)
        mask[mask != 0] = 1
        mask = np.array(mask, dtype=np.uint8)
        return mask

    def _get_features_columns(self):
        mask = self._load_mask(self.rnd_mask_pth)
        region_prop_table = regionprops_table(mask, properties=self.features_list)
        columns = region_prop_table.keys()
        if self.print_:
            print("Feature columns provided.")
        return columns

