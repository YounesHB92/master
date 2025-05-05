import os
import sys
from pathlib import Path

# Set working directory to one level up from this file's location
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
print("Current working directory set to:", os.getcwd())

from src.utils import find_configs, load_env_variables, send_sms

env = load_env_variables()
from src.datasets import Splitter, DatasetIterator, DatasetLoader
from src.training import Trainer, LossAndMetrics
from src.models import LoadModel
from datetime import datetime
import torch
import warnings
import yaml
import pytz

time_zone = pytz.timezone("Australia/Brisbane")

warnings.filterwarnings("ignore")

print("Loading environment variables for:", env)

for dir_ in ["CHECKPOINTS_DIR", "LOGS_DIR"]:  # make the output folders if they do not exist
    os.makedirs(os.getenv(dir_), exist_ok=True)
print("Output folders created.")

configs = find_configs()
print("Number of configurations found: ", len(configs.keys()))

for config_name in configs.keys():
    print(f"Configuration: {config_name}")

    config = configs[config_name]

    # saving the config file
    config_save_name = f"{config_name}_{datetime.now(time_zone).strftime('%Y-%m-%d_%H-%M-%S')}"

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
        config_name=config_save_name,
        val_loader=val_loader,
        device="cuda",
        **config["trainer"]
    )
    trainer_.train()

    with open(os.path.join(os.getenv("CHECKPOINTS_DIR"), config_save_name + ".yaml"), "w") as file:
        yaml.dump(config, file)

    sms = f"""
    Hey Boss,
    Environment: {env}
    Training done for {config_save_name}, checkpoints, logs and config files are saved in the output folders.
    """

    send_sms(
        sid=os.getenv("TWILIO_SID"),
        token=os.getenv("TWILIO_TOKEN"),
        from_=os.getenv("TWILIO_FROM"),
        to=os.getenv("TWILIO_TO"),
        message=sms,
    )
