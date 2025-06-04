import os

from src.utils import load_env_variables
from src.utils import print_indented

_ = load_env_variables()
import shutil
import src.utils as utils
from sklearn.model_selection import train_test_split


class SplitterCore:
    def __init__(self, test_val_ratio, force_dir, random_state=42, *args, **kwargs):
        self.test_val_ratio = test_val_ratio
        self.force_dir = force_dir
        self.random_state = random_state

        self.set_names = ["train", "val", "test"]
        self.raw_images_path = None
        self.raw_masks_path = None
        self._check_raw_files()

        self.sets = None
        self._run()

        super().__init__(*args, **kwargs)

    def _check_raw_files(self):
        images_raw_path = os.path.join(os.getenv("RAW_DATA_DIR"), "images")
        masks_raw_path = os.path.join(os.getenv("RAW_DATA_DIR"), "masks")
        for path in [images_raw_path, masks_raw_path]:
            print("\nChecking raw path: ", path)
            files = os.listdir(path)
            for file in utils.tqdm_print(files, desc="Checking files", total=len(files)):
                if not file.endswith(('.jpg', '.png', '.jpeg')):
                    raise ValueError(f"Invalid file format: {file}. Only .jpg, .png, and .jpeg are allowed.")
            print_indented("All raw files are val.", level=1)
        self.raw_images_path = images_raw_path
        self.raw_masks_path = masks_raw_path

    def _run(self):
        print("\nChecking split situations")
        splits_dir = os.getenv("SPLIT_DATA_DIR")
        mode = None
        if len(os.listdir(splits_dir)) == 0:
            print_indented("Split directory is empty. Creating new directory.", level=1)
            mode = "fresh"
        elif len(os.listdir(splits_dir)) > 0:
            print_indented(f"Split directories are there. Force Directory -> {self.force_dir}", level=1)
            if self.force_dir:
                self._delete_sets_dirs()
                print_indented("Split dirs are removed", level=2)
                mode = "fresh"
            elif not self.force_dir:
                print_indented("Train, val, test will be fetched from current directory.", level=2)
                mode="read_existing"
        self._train_val_test_split(mode=mode)

    def _copy_images_masks(self, set_name, set_data):
        print("\nCopying images and masks for set: ", set_name)
        for image_file in utils.tqdm_print(set_data["images"], desc="Copying images", total=len(set_data["images"])):
            source_path = os.path.join(self.raw_images_path, image_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images", image_file)
            shutil.copyfile(source_path, destination_path)
        for mask_file in utils.tqdm_print(set_data["masks"], desc="Copying masks", total=len(set_data["masks"])):
            source_path = os.path.join(self.raw_images_path, mask_file)
            destination_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks", mask_file)
            shutil.copyfile(source_path, destination_path)

    def _create_sets_dirs(self):
        for set_name in self.set_names:
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "images"))
            os.mkdir(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name, "masks"))

    def _delete_sets_dirs(self):
        for set_name in self.set_names:
            shutil.rmtree(os.path.join(os.getenv("SPLIT_DATA_DIR"), set_name))

    def _copy_sets(self):
        for set_name, set_data in self.sets.items():
            self._copy_images_masks(set_name, set_data)

    def _train_val_test_split(self, mode):
        if mode == "fresh":
            self._create_sets_dirs()

            images_files = os.listdir(self.raw_images_path)
            masks_files = os.listdir(self.raw_masks_path)

            x_train, x_test, y_train, y_test = train_test_split(images_files, masks_files,
                                                                test_size=self.test_val_ratio,
                                                                random_state=self.random_state)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.test_val_ratio,
                                                              random_state=self.random_state)
            self.sets = {
                "train": {
                    "images": [file for file in x_train],
                    "masks": [file for file in y_train]
                },
                "val": {
                    "images": [file for file in x_val],
                    "masks": [file for file in y_val]
                },
                "test": {
                    "images": [file for file in x_test],
                    "masks": [file for file in y_test]
                }
            }
            self._copy_sets()
        elif mode == "read_existing":
            self.sets = {}
            for set_ in self.set_names:
                images_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_, "masks")
                masks_path = os.path.join(os.getenv("SPLIT_DATA_DIR"), set_, "images")
                self.sets[set_] = {
                    "images": [file for file in os.listdir(images_path)],
                    "masks": [file for file in os.listdir(masks_path)]
                }