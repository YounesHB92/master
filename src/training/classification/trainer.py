import os

from sklearn.preprocessing import StandardScaler

from src.datasets import FeatureLoader
from src.models.classification import LoadModels
from src.utils import load_env_variables, print_indented
from pprint import pprint

_ = load_env_variables()

class Trainer(FeatureLoader, LoadModels):
    def __init__(self, features_file):
        super().__init__(features_file)
        self.check()

    def check(self):
        if self.features is None:
            raise Exception("Features not loaded")
        if self.models_all is None:
            raise Exception("Models not loaded")

    def train(self):
        self.check()
        self.drop_columns()

        print("\nFeatures used for training:")
        pprint(list(self.features["x_train"].columns))

        self.scale_features()

        x_train = self.features["x_train_scaled"]
        y_train = self.features["y_train"].values.ravel()
        x_val = self.features["x_val_scaled"]
        y_val = self.features["y_val"].values.ravel()



        results = {}

        for model_name, model_ in self.models_all.items():
            print("\nTraining model", model_name)
            results[model_name] ={}
            model_.fit(x_train, y_train)
            results[model_name]["train_accuracy"] = model_.score(x_train, y_train)
            results[model_name]["val_accuracy"] = model_.score(x_val, y_val)

        return results

    def drop_columns(self):
        x_columns_to_drop = [
            "image",
            "set"
        ]
        y_columns_to_drop = [
            "image",
            "set"
        ]

        for key, item in self.features.items():
            if key.find("x") > -1:
                self.features[key] = self.features[key].drop(x_columns_to_drop, axis=1, errors="ignore")
            if key.find("y") > -1:
                self.features[key] = self.features[key].drop(y_columns_to_drop, axis=1, errors="ignore")
        print("Feature area ready to be trained!")

    def report(self, results):
        for model_name, results in results.items():
            print(f"\nModel name: {model_name}")
            print_indented(f"Train Accuracy: {results['train_accuracy']*100:.2f}")
            print_indented(f"Val Accuracy: {results['val_accuracy']*100:.2f}")

    def scale_features(self):
        scaler = StandardScaler()
        scaler.fit(self.features["x_train"])
        self.features["x_train_scaled"] = scaler.transform(self.features["x_train"])
        self.features["x_val_scaled"] = scaler.transform(self.features["x_val"])
        print("\nScaling has been done!")

