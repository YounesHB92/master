import os
import sys
from pathlib import Path

# Set working directory to one level up from this file's location
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
print("Current working directory set to:", os.getcwd())

from src.utils import find_configs, load_env_variables, send_sms
base_path, env = load_env_variables()
from src.datasets import CnnSplitter, CnnDatasetIterator, CnnDatasetLoader
from src.models import CnnModel
from src.training import CnnTrainer, CnnLossAndMetrics
import pytz
from datetime import datetime
time_zone = pytz.timezone("Australia/Brisbane")
import yaml
import warnings
warnings.filterwarnings("ignore")

print("Loading environment variables for:", env)

mode = "cnn"
for dir_ in ["CHECKPOINTS_DIR", "LOGS_DIR"]:  # make the output folders if they do not exist
    os.makedirs(os.path.join(os.getenv(dir_), mode), exist_ok=True)
print("Output folders created.")

configs = find_configs(mode=mode)
print("Number of configurations found: ", len(configs.keys()))

for config_name in configs.keys():
    print(f"Configuration: {config_name}")

    config = configs[config_name]
    # saving the config file
    config_save_name = f"{config_name}_{datetime.now(time_zone).strftime('%Y-%m-%d_%H-%M-%S')}"

    splitter_ = CnnSplitter(
        **config["cnn_splitter"]
    )

    train_iterator_ = CnnDatasetIterator(
        **config["datasets"]["train"]["iterator"]
    )

    val_iterator_ = CnnDatasetIterator(
        **config["datasets"]["val"]["iterator"]
    )

    train_dataset_ = CnnDatasetLoader(
        dataset=train_iterator_,
        **config["datasets"]["train"]["loader"]
    )
    train_loader = train_dataset_.loader

    val_dataset_ = CnnDatasetLoader(
        dataset=val_iterator_,
        **config["datasets"]["val"]["loader"]
    )
    val_loader = val_dataset_.loader

    model_ = CnnModel(
        encoder_name=config["model"]["encoder_name"],
        num_classes=len(train_iterator_.classes_list),
    )

    loss_and_metrics_ = CnnLossAndMetrics(
        num_classes=len(train_iterator_.classes_list),
    )

    trainer_ = CnnTrainer(
        model=model_,
        train_dataset=train_loader,
        val_dataset=val_loader,
        loss_and_metrics=loss_and_metrics_,
        config_name=config_save_name,
        **config["trainer"]
    )
    trainer_.train()

    with open(os.path.join(os.getenv("CHECKPOINTS_DIR"), mode, config_save_name + ".yaml"), "w") as file:
        yaml.dump(config, file)

    sms = f"""
    Hey Boss,
    Environment: {env}
    Mode: CNN
    Training done for {config_save_name}, checkpoints, logs and config files are saved in the output folders.
    """

    send_sms(
        sid=os.getenv("TWILIO_SID"),
        token=os.getenv("TWILIO_TOKEN"),
        from_=os.getenv("TWILIO_FROM"),
        to=os.getenv("TWILIO_TO"),
        message=sms,
    )