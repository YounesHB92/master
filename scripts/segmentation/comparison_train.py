import os
import sys
from pathlib import Path

# Set working directory to one level up from this file's location
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))
print("Current working directory set to:", os.getcwd())

from src.utils import time_, env_
from src.datasets import SegmentationIterator, SegmentationDatasetLoader
from src.models import SegmentationModel
from src.training import SegmentationLossAndMetrics, SegmentationTrainer
from src.utils import vars
from torch import optim

working_env = env_.find_env()
print("Loading environment variables for:", env_)

checkpoints_dir = env_.get_checkpoints_path()
logs_dir = env_.get_logs_path()

for path_ in [checkpoints_dir, logs_dir]:
    os.makedirs(path_, exist_ok=True)

configs = env_.find_configs(mode="segmentation", env=working_env)
print("Number of configurations found: ", len(configs.keys()))

for config_name, configs in configs.items():
    db_name = configs["db_name"]

    config_save_name = config_name + "_" + db_name
    config_save_name = time_.add_time(config_save_name)

    train_iterator_ = SegmentationIterator(
        db_name=db_name,
        set_name="train",
        patch_size=configs["patch_size"],
        augment=True
    )

    val_iterator_ = SegmentationIterator(
        db_name=db_name,
        set_name="val",
        patch_size=configs["patch_size"],
        augment=False
    )

    train_dataset = SegmentationDatasetLoader(
        dataset=train_iterator_,
        classes=vars.datasets_classes[db_name],
        batch_size=4,
        num_workers=4,
        show_samples=False
    )
    train_loader = train_dataset.loader

    val_dataset = SegmentationDatasetLoader(
        dataset=val_iterator_,
        classes=vars.datasets_classes[db_name],
        batch_size=4,
        num_workers=4,
        show_samples=False
    )
    val_loader = val_dataset.loader

    seg_model_ = SegmentationModel(
        model_name=configs["model_name"],
        encoder_name=configs["encoder_name"],
        encoder_weights="imagenet",
        num_classes=len(vars.datasets_classes[db_name])
    )

    model_ = seg_model_.load_model()

    loss_and_metrics = SegmentationLossAndMetrics(
        num_classes=len(vars.datasets_classes[db_name])
    )
    optimizer = optim.Adam(model_.parameters(), lr=1e-3)

    trainer_ = SegmentationTrainer(
        model=model_,
        loss_metrics=loss_and_metrics,
        device="cuda",
        train_loader=train_loader,
        optimizer=optimizer,
        val_loader=val_loader,
        epochs=configs["epochs"],
        config_name=config_save_name,
    )
    trainer_.train()
