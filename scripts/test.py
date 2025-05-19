from src.datasets.segmentation import SegmentationSplitter as SegmentationSplitter
from pprint import pprint

seg_splitter = SegmentationSplitter(
    test_val_ratio=0.2,
    force_directory=False,
    random_state=42,
    classes_list=["background", "crack"],
    rebuild_masks=True
)