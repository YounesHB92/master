import os

import torch
from PIL import Image


class Predictor:
    def __init__(self, model, dataset_loader, device, output_dir=None, folder_name=None):
        self.model = model
        self.transformer = dataset_loader.val_transformer
        self.train_dataset = dataset_loader.train_dataset  # to extract number of classes
        self.device = device
        self.output_dir = output_dir
        self.folder_name = folder_name
        self.report()

    def predict_all(self, images_path):
        print(f"Checking path: {images_path}")

        self.classes = self.train_dataset.num_classes
        for class_ in self.classes:
            path_ = os.path.join(images_path, class_)
            for image_ in os.listdir(path_):
                if image_.split(".")[-1] not in ["jpg", "jpeg", "png"]:
                    raise ValueError(f"Invalid image type {os.path.join(path_, image_)}")

        if self.output_dir is None or self.folder_name is None:
            raise Exception("Output directory or Sub directory is not specified!")

        # creating directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, self.folder_name), exist_ok=True)

        self.results = {
            "image": [],
            "label": [],
            "prediction": []
        }
        for class_ in self.classes:
            print("Predicting images for class {}".format(class_))
            path_ = os.listdir(os.path.join(images_path, class_))
            for image_ in os.listdir(os.path.join(images_path, class_)):
                print("\tPredicting images for image {}".format(image_))
                self.results["image"].append(image_)
                self.results["label"].append(class_)
                prediction = self.predict_image(os.path.join(images_path, class_, image_))
                self.results["prediction"].append(prediction)

    def predict_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = self.transformer(image).unsqueeze(0).to(self.device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)

        return self.train_dataset.num_classes[predicted.item()]

    def report(self):
        print("Predictor is ready!\n\npredictor_.predict_image to be used.")
