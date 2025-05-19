import os
from src.features import FeatureLoader
from src.utils import load_env_variables
_ = load_env_variables()

class Trainer(FeatureLoader):
    def __init__(self, features_file):
        super().__init__(features_file)

    def train_val_test(self):
        split_dir = os.getenv("SPLIT_DATA_DIR")
        splits = {}
        for set_ in os.listdir(split_dir):
            path_ = os.path.join(split_dir, set_, "masks")
            splits[set_] = os.listdir(path_)

        return splits

