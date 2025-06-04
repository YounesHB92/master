import os

import pandas as pd
import yaml
from dotenv import load_dotenv

def find_env():
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"


def load_env_variables():
    environment = find_env()
    if environment == "colab":
        base_path = "/content/drive/MyDrive/projects/master"
    else:
        base_path = "/home/younes/Desktop/projects/master"
    dotenv_path = os.path.join(base_path, ".env." + environment)
    if not os.path.exists(dotenv_path):
        raise Exception(f"Environment file not found at: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    return base_path, environment

_, _ = load_env_variables()


def find_configs(mode, env):
    if env == "colab":
        configs_file = f"/content/master/configs/colab_{mode}_configs.yaml"
    else:
        configs_file = f"configs/local_{mode}_configs.yaml"
    with open(configs_file, "r") as file:
        configs = yaml.safe_load(file)
    return configs


def get_config_info(config_path, mode):
    config_name = os.path.basename(config_path)
    config_dir = os.path.dirname(config_path)
    # finding the respective model
    model_path = None
    for file in os.listdir(config_dir):
        if file.endswith('.pt') and file.split(".")[0] == config_name.split(".")[0]:
            model_path = os.path.join(config_dir, file)

    if model_path is None:
        raise Exception("Respective model not found for config file: ", config_name)

    # extracting date and time from the config name
    config_name_pure = config_name.split(".")[0]
    config_name_split = config_name_pure.split("_")
    date = config_name_split[-2]
    time = config_name_split[-1]

    # load log files
    experiments_path = os.path.dirname(os.path.dirname(os.path.dirname(config_path)))
    logs_path = os.path.join(experiments_path, "logs", mode)
    log_file_name = config_name.replace(".yaml", ".csv")
    logs = pd.read_csv(os.path.join(logs_path, log_file_name))

    # load config file itself
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if mode == "segmentation":
        classes = config["splitter"]["classes_list"]
        trained_model = config["model"]["model_name"]
    else:
        classes = None
        trained_model = None
    patch_size = config["datasets"]["train"]["iterator"]["patch_size"]
    encoder = config["model"]["encoder_name"]

    config_info = {
        "model_path": model_path,
        "date": date,
        "time": time,
        "logs": logs,
        "classes": classes,
        "patch_size": patch_size,
        "trained_model": trained_model,
        "encoder": encoder
    }

    return config_info

def get_split_path(db_name="mine"):
    return os.path.join(os.getenv("SPLIT_DATA_DIR"), db_name)

def get_raw_path():
    return os.getenv("RAW_DATA_DIR")

def get_working_dir(env_name):
    if env_name == "local":
        return "/home/younes/Desktop/projects/master"
    elif env_name == "colab":
        return "/content/master"
    else:
        raise Exception(f"Unknown environment name: {env_name}")

def get_checkpoints_path():
    return os.getenv("CHECKPOINTS_DIR")

def get_logs_path():
    return os.getenv("LOGS_DIR")