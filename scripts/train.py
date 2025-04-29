import src.utils as utils
from src.datasets import Splitter, DatasetIterator, DatasetLoader
from src.training import Trainer, LossAndMetrics
from src.models import LoadModel
import torch
import yaml
import warnings
warnings.filterwarnings("ignore")

env = utils.find_env()
print("Environment detected:", env)

configs_file = f"{env}_configs.yaml"
with open(configs_file, "r") as file:
    configs = yaml.safe_load(file)

print("Number of configurations found: ", len(configs.keys()))
for config_name in configs.keys():
    print(f"Configuration: {config_name}")
    print("Configurations loaded for environment:", env)

    config = configs[config_name]
    splitter_ = Splitter(**config["splitter"])
    train_iterator_ = DatasetIterator(**config["datasets"]["train"]["iterator"])
    val_iterator_ = DatasetIterator(**config["datasets"]["val"]["iterator"])

    train_loader_ = DatasetLoader(
        dataset=train_iterator_,
        classes=splitter_.classes,
        **config["datasets"]["train"]["loader"]
    )
    train_loader = train_loader_.loader

    val_loader_ = DatasetLoader(
        dataset=val_iterator_,
        classes=splitter_.classes,
        **config["datasets"]["val"]["loader"]
    )
    val_loader = val_loader_.loader

    model_loader_ = LoadModel(
        model_name=config["model"]["model_name"],
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=config["model"]["encoder_weights"],
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
        config_name=config_name,
        val_loader=val_loader,
        device="cuda",
        **config["trainer"]
    )

    trainer_.train()