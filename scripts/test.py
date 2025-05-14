import os

import torch
import yaml

from src.datasets import DatasetLoader, DatasetIterator, Splitter
from src.models import LoadModel
from src.testing import Tester
from src.training import LossAndMetrics
from src.utils import load_env_variables

_ = load_env_variables()

checkpoints_dir = os.getenv('CHECKPOINTS_DIR')
models = [file for file in os.listdir(checkpoints_dir) if file.endswith(".pt")]
print("Total models state dicts found: ", len(models))

state_dict_path = os.path.join(checkpoints_dir, models[0])

config_path = state_dict_path.replace('.pt', '.yaml')
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

config["splitter"]["force_directory"] = False  #### temporary set to False

splitter_ = Splitter(**config["splitter"])

test_iterator_ = DatasetIterator(
    set_name="test",
    patch_size=config["datasets"]["val"]["iterator"]["patch_size"],
    augment=False
)

test_loader_ = DatasetLoader(
    dataset=test_iterator_,
    classes=splitter_.classes,
    **config["datasets"]["val"]["loader"]
)
test_loader = test_loader_.loader

loss_and_metrics_ = LossAndMetrics(
    num_classes=len(splitter_.classes)
)

model_loader_ = LoadModel(
    num_classes=len(splitter_.classes),
    **config["model"]
)
model = model_loader_.load_model()
model.load_state_dict(torch.load(state_dict_path))

tester_ = Tester(
    model=model,
    loss_metrics=loss_and_metrics_,
    device="cuda",
    test_loader=test_loader
)
tester_.test()
