import os
from . import  vars
import numpy as np
from PIL import Image
import cv2

import random

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


def label_to_mask(image_path, label_path):
    # Load image
    img = cv2.imread(image_path)
    annotated_img = img.copy()
    h, w = img.shape[:2]

    # Initialize mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fixed color map (10 distinct colors)
    class_colors = vars.class_colors

    # Read label file
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                parts = line.strip().split()
                raw_class_id = parts[0]
                if raw_class_id.lower() == "none":
                    continue  # skip line

                class_id = int(float(raw_class_id)) + 1  # reserve 0 for background
                coords = list(map(float, parts[1:]))

                points = np.array([[int(x * w), int(y * h)] for x, y in zip(coords[::2], coords[1::2])], dtype=np.int32)

                cv2.fillPoly(mask, [points], color=class_id)

                # Draw outline and label in annotated image
                color = class_colors.get(class_id, (255, 255, 255))  # fallback to white
                cv2.polylines(annotated_img, [points], isClosed=True, color=color, thickness=2)
                cv2.putText(annotated_img, f"{class_id}", tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            except Exception as e:
                print(f"[Warning] Skipping bad line in {label_path}: '{line.strip()}' ({e})")
                continue  # skip malformed or bad lines

    return mask, annotated_img

def visualize_mask(mask):
    # Define same fixed class colors (BGR turned to RGB for visualization)
    class_colors = vars.class_colors

    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color

    return rgb_mask