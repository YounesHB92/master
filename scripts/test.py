import os
from src.utils import load_env_variables
import re
load_env_variables()

checkpoints_dir = os.getenv('CHECKPOINTS_DIR')
configs_files = [file for file in os.listdir(checkpoints_dir) if file .endswith('.yaml')]
configs_paths = [os.path.join(checkpoints_dir, file) for file in configs_files]

config_path = configs_paths[0]

config_name = os.path.basename(config_path)
config_dir = os.path.dirname(config_path)
# finding the respective model
for file in os.listdir(config_dir):
    if file.endswith('.pt') and file.split(".")[0] == config_name.split(".")[0]:
        print("Respective model found: ", file)
        model_path = os.path.join(config_dir, file)

# extracting date and time from the config name
config_name_pure = config_name.split(".")[0]
config_name_split = config_name_pure.split("_")
date = config_name_split[-2]
print("Date found: ", date)
time = config_name_split[-1]
print("Time found: ", time)