from src.datasets import CnnSplitter, CnnDatasetIterator, CnnDatasetLoader
from src.models import CnnModel
from src.training import CnnTrainer, CnnLossAndMetrics

splitter_ = CnnSplitter(
    test_val_ratio=0.2,
    force_dir=False,
    force_subdir=False,
    random_state=42
)

train_iterator_ = CnnDatasetIterator(
    set_name="train",
    patch_size=512,
    augment=True
)

val_iterator_ = CnnDatasetIterator(
    set_name="val",
    patch_size=512,
    augment=False
)

train_dataset_ = CnnDatasetLoader(
    dataset=train_iterator_,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    report=True
)
train_loader = train_dataset_.loader

val_dataset_ = CnnDatasetLoader(
    dataset=val_iterator_,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    report=True
)
val_loader = val_dataset_.loader

model_ = CnnModel(
    encoder_name="resnet18",
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
    epochs=1
)
trainer_.train()
