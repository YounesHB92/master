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


def find_configs(mode):
    env = find_env()
    if env == "colab":
        configs_file = f"/content/master/colab_{mode}_configs.yaml"
    else:
        configs_file = f"local_{mode}_configs.yaml"
    with open(configs_file, "r") as file:
        configs = yaml.safe_load(file)
    return configs


def get_config_info(config_path):
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
    experiments_path = os.path.dirname(os.path.dirname(config_path))
    logs_path = os.path.join(experiments_path, "logs")
    log_file_name = config_name.replace(".yaml", ".csv")
    logs = pd.read_csv(os.path.join(logs_path, log_file_name))

    # load config file itself
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    classes = config["splitter"]["classes_list"]
    patch_size = config["datasets"]["train"]["iterator"]["patch_size"]
    trained_model = config["model"]["model_name"]
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
