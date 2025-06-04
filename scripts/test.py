from torch import optim

from src.datasets import SegmentationIterator, SegmentationDatasetLoader
from src.models import SegmentationModel
from src.training import SegmentationLossAndMetrics, SegmentationTrainer
from src.utils import vars


db_name = "ppdd"
train_iterator_ = SegmentationIterator(
    db_name=db_name,
    set_name="train",
    patch_size=256,
    augment=True
)

val_iterator_ = SegmentationIterator(
    db_name=db_name,
    set_name="val",
    patch_size=256,
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
    model_name="DeepLabV3Plus",
    encoder_name="resnet50",
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
    epochs=2,
    config_name="test_local"
)
trainer_.train()