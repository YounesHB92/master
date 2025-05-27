from src.datasets import CnnDatasetIterator, CnnDatasetLoader

iterator_ = CnnDatasetIterator(
    set_name="train",
    patch_size=256,
    augment=True
)

loader_ = CnnDatasetLoader(
    dataset=iterator_,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    report=True
)