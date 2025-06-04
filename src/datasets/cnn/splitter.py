import os
import shutil

from src.utils import env_, path_, general
from src.datasets import SplitterCore
from tqdm import tqdm

class CnnSplitter(SplitterCore):
    """
    CNN Dataset Splitter
    Args:
        test_val_ratio: test val ratio
        force_dir: if set to true the current split dir will be overwritten (used for the main train, val, test sets)
        random_state: random seed (defaults to 42)
        force_subdir: if set to true the train, val, test dirs containing the classes will be overwritten
    Attributes:
        self.classes: pd.Dataframe containing image, type, set columns
    See Also:
        SplitterCore
    """
    def __init__(self, test_val_ratio, force_dir, random_state=42, force_subdir=False, *args, **kwargs):
        super().__init__(test_val_ratio, force_dir, random_state, *args, **kwargs)

        self.split_dir = env.get_split_path()
        self.output_dir = os.path.join(os.getenv("SPLIT_DATA_DIR"), "cnn_splits")
        self.force_subdir = force_subdir
        path_.handle_path(self.output_dir, force=force_subdir)


        self.load_classes()
        self.run()


    def load_classes(self):
        self.classes = general.load_classes()
        if "type" not in self.classes:
            raise Exception("<type> could not be found. Exiting...")

        # here we add sets to the self.classes
        self.classes = general.add_set_column(self.classes, self.sets)

    def run(self):
        print("\nCNN splitting")
        if self.sets is None:
            raise Exception("self.sets is None. Cannot run without sets")

        if self.force_subdir:
            unique_types = list(self.classes["type"].unique())
            unique_sets = self.set_names
            for set_name in unique_sets:
                os.makedirs(os.path.join(self.output_dir, set_name), exist_ok=True)
                for unique_type in unique_types:
                    os.makedirs(os.path.join(self.output_dir, set_name, unique_type), exist_ok=True)

            print("Directories created successfully. Getting ready to transfer the files")
            loop = tqdm(self.classes.index, total=len(self.classes.index), desc="CNN splitting")

            source_base_path = os.getenv("RAW_DATA_DIR")
            for class_index in loop:
                image_file = self.classes.loc[class_index, "image"]
                type_ = self.classes.loc[class_index, "type"]
                set_ = self.classes.loc[class_index, "set"]

                source_path = os.path.join(source_base_path, "masks", image_file)
                destination_path = os.path.join(self.output_dir, set_, type_, image_file)
                shutil.copy(source_path, destination_path)

            print("All image files are transferred!")
        else:
            print("self.force_subdir -> False, skipping...")