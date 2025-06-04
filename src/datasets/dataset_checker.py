import os

from src.utils import env_, vars, path_


class DatasetChecker:
    def __init__(self, db_name):
        self.db_name = db_name
        self.split_dir = env.get_split_path(db_name)
        self.check_existence()
        self.check_images()

    def check_existence(self):
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Dataset {self.db_name} does not exist")

    def check_images(self):
        print("\nChecking images files sanity...")
        for set_name in vars.set_names:
            path = os.path.join(self.split_dir, set_name, "images")
            path_.check_image_files(path)
        print("\tAll images are val")
        self.report()

    def report(self):
        for set_name in vars.set_names:
            print(f"\n{set_name.upper()} report:")
            set_path = os.path.join(self.split_dir, set_name)
            dirs = os.listdir(set_path)
            print(f"\t{len(dirs)} dirs found.")
            print("\t\t" + ", ".join(dirs))