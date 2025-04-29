import src.utils as utils
from src.datasets import Splitter, DatasetIterator, DatasetLoader
from src.training import Trainer, LossAndMetrics
from src.models import LoadModel
import torch
import yaml
import warnings
from src.utils import print_indented
warnings.filterwarnings("ignore")

env = utils.find_env()
print("Environment detected:", env)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

configs = config["envs"][env]
print("Configurations loaded for environment:", env)

splitter_ = Splitter(**configs["splitter"])
train_iterator_ = DatasetIterator(**configs["datasets"]["train"]["iterator"])
val_iterator_ = DatasetIterator(**configs["datasets"]["val"]["iterator"])

train_loader_ = DatasetLoader(
    dataset=train_iterator_,
    classes=splitter_.classes,
    **configs["datasets"]["train"]["loader"]
)
train_loader = train_loader_.loader

val_loader_ = DatasetLoader(
    dataset=val_iterator_,
    classes=splitter_.classes,
    **configs["datasets"]["val"]["loader"]
)
val_loader = val_loader_.loader

model_loader_ = LoadModel(
    model_name=configs["model"]["model_name"],
    encoder_name=configs["model"]["encoder_name"],
    encoder_weights=configs["model"]["encoder_weights"],
    num_classes=len(splitter_.classes)
)

model = model_loader_.load_model()
print("Model loaded successfully.")

loss_and_metrics_ = LossAndMetrics(num_classes=len(splitter_.classes))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer_ = Trainer(
    model=model,
    optimizer=optimizer,
    loss_metrics=loss_and_metrics_,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda"
)

trainer_.train(epochs=2)