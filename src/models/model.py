import torch

from src.models.classification_head import ClassificationHead


class Model:
    def __init__(self, encoder, num_classes):
        self.encoder = encoder
        self.num_classes = num_classes
        self.load_head()
        self.get_model_ready()
        self.report()

    def load_head(self):
        self.head_ = ClassificationHead(
            self.encoder,
            self.num_classes
        )

    def get_model_ready(self):
        self.model = self.head_.to("cuda" if torch.cuda.is_available() else "cpu")

    def report(self):
        print(self.model)
        print("\nModel is ready!\n\t model_.model to be used!")
