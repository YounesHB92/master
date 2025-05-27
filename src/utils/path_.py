import os
import shutil

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
