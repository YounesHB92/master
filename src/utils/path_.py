import os
import shutil

from tqdm import tqdm
from . import env_


def handle_path(path, force=False):
    if not os.path.exists(path):
        print("Path is not found. Creating new path at: {}".format(path))
        os.makedirs(path)
    else:
        if force:
            print("Path already exists. Overwriting due to FORCE")
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print("Path already exists. Force not set. Skipping.")
            pass


def check_image_files(path):
    files = os.listdir(path)
    loop = tqdm(files, total=len(files), desc="Checking image files in {}".format(path))
    for file in loop:
        if file.split(".")[-1] not in ["jpg", "jpeg", "png", "PNG"]:
            raise ValueError("Image must be JPEG or PNG, found {}".format(file))

def change_working_dir():
    env_ = env.find_env()
    working_dir = env.get_working_dir(env_name=env_)
    os.chdir(working_dir)
    print("Current working dir: {}".format(working_dir))
