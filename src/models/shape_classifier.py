import os
import pandas as pd

from src.utils import load_env_variables, print_indented
_ = load_env_variables()


class ShapeClassifier:
    def __init__(self, features_file):
        self.features_path = os.getenv("FEATURES_DIR")
        self.features_file = features_file
        self.features_df = self.load_features()
        self.classes_df = self.load_classes()

    def load_features(self):
        features_path = os.path.join(self.features_path, self.features_file)
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        features = pd.read_csv(features_path)
        print("Features loaded from: {}".format(self.features_file))
        print_indented("Number of features: {}".format(len(features.columns)), level=1)
        print_indented("Number of samples: {}".format(len(features)), level=1)
        return features

    def load_classes(self):
        classes_path = os.path.join(os.getenv("RAW_DATA_DIR"), "classes", "crack_types.csv")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        classes = pd.read_csv(classes_path)
        print("Classes loaded from: {}".format(os.path.basename(classes_path)))
        print_indented("Number of columns: {}".format(len(classes.columns)), level=1)
        print_indented("Number of rows: {}".format(len(classes.index)), level=1)
        return classes

    def do_the_checks(self):
        pass
