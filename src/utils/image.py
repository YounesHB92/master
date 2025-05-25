import os

import numpy as np
from PIL import Image

from tqdm import tqdm


def check_image_files(path, verbose=True):
    files = os.listdir(path)
    loop = tqdm(files, total=len(files), desc="Checking image files in {}".format(path))
    for file in loop:
        if file.split(".")[-1] not in ["jpg", "jpeg", "png"]:
            raise ValueError("Image must be JPEG or PNG, found {}".format(file))
    if verbose:
        print("All image files are valid, path: {}".format(path))


def rebuild_mask(mask_file, classes_list, classes_dict, crack_types):
    # load_mask
    raw_masks_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
    mask_path = os.path.join(raw_masks_path, mask_file)
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    rebuilt_mask = np.zeros_like(mask, dtype=np.uint8)
    if len(classes_list) == 2:
        rebuilt_mask[mask != 0] = 1
        return rebuilt_mask.astype("uint8")
    else:
        index_ = crack_types[crack_types["image"] == mask_file].index
        crack_type = crack_types.loc[index_, "type"].values[0]
        if crack_type not in classes_dict.keys():
            crack_type = "other"
        rebuilt_mask[mask != 0] = classes_dict[crack_type]
        return rebuilt_mask.astype("uint8")


def save_rebuilt_mask(rebuilt_mask, path):
    Image.fromarray(rebuilt_mask).save(path)
