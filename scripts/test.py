import os
from PIL import Image
from tqdm import tqdm

# Set your paths here
RAW_IMAGES_PATH = "./datasets/raw/images"
RAW_MASKS_PATH = "./datasets/raw/masks"

# Optional: set True to overwrite masks, False to save fixed versions separately
OVERWRITE = False

# Optional: where to save corrected masks
FIXED_MASKS_PATH = "./datasets/raw/masks_fixed"

if not OVERWRITE:
    os.makedirs(FIXED_MASKS_PATH, exist_ok=True)

image_files = sorted(os.listdir(RAW_IMAGES_PATH))
mask_files = sorted(os.listdir(RAW_MASKS_PATH))

assert len(image_files) == len(mask_files), "Mismatch between number of images and masks!"

for img_name, mask_name in tqdm(zip(image_files, mask_files), total=len(image_files)):
    img_path = os.path.join(RAW_IMAGES_PATH, img_name)
    mask_path = os.path.join(RAW_MASKS_PATH, mask_name)

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img_size = img.size  # (W, H)
    mask_size = mask.size  # (W, H)

    if img_size != mask_size:
        print(f"Mismatch found in {img_name} / {mask_name}: Image {img_size}, Mask {mask_size}")

        # Try rotating mask to fix it
        mask_rotated = mask.rotate(90, expand=True)
        if mask_rotated.size == img_size:
            mask = mask_rotated
            print(f"Fixed by rotating mask 90°: {mask_name}")
        else:
            # Try 270 degrees just in case
            mask_rotated = mask.rotate(270, expand=True)
            if mask_rotated.size == img_size:
                mask = mask_rotated
                print(f"Fixed by rotating mask 270°: {mask_name}")
            else:
                print(f"❌ Could not fix {mask_name} by rotation. Please check manually.")
                continue  # Skip

    # Save the fixed (or unchanged) mask
    if OVERWRITE:
        mask.save(mask_path)
    else:
        img.save(os.path.join(FIXED_MASKS_PATH, "image_" + img_name))
        mask.save(os.path.join(FIXED_MASKS_PATH, "mask_"+ mask_name))

print("\nFinished checking and fixing masks!")
