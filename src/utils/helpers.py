import os
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from twilio.rest import Client
import matplotlib.pyplot as plt
import seaborn as sns


def find_encoder_name(model_name):
    parts = model_name.split("_")
    encoder_index = None
    epoch_index = None
    for num, part_ in enumerate(parts):
        if part_ == "encoder":
            encoder_index = num
        elif part_ == "epoch":
            epoch_index = num

    if encoder_index is None or epoch_index is None:
        raise Exception("Could not find a valid encoder name")

    encoder_name = parts[encoder_index + 1:epoch_index]
    return "_".join(encoder_name)


def find_env():
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"


def inspect_object(obj):
    print(f"Inspecting object: {obj.__class__.__name__}")
    for attr, value in vars(obj).items():
        print(f"{attr}: {value}")


def tqdm_print(*args, **kwargs):
    return tqdm(*args, file=sys.stdout, **kwargs)


def print_indented(text, level=1):
    indent = "\t" * level
    print(indent + text)


def load_env_variables():
    environment = find_env()
    if environment == "colab":
        base_path = "/content/master"
    else:
        base_path = "/home/younes/Desktop/projects/master"
    dotenv_path = os.path.join(base_path, ".env." + environment)
    if not os.path.exists(dotenv_path):
        raise Exception(f"Environment file not found at: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    return base_path, environment


def find_configs():
    env = find_env()
    if env == "colab":
        configs_file = "/content/drive/MyDrive/projects/master/colab_configs.yaml"
    else:
        configs_file = "local_configs.yaml"
    with open(configs_file, "r") as file:
        configs = yaml.safe_load(file)
    return configs


def send_sms(message, sid, token, from_, to):
    account_sid = sid
    auth_token = token
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_=from_,
        to=to,
        body=message)
    print(message.sid)


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


def plot_log_info(config_info):
    train_model = config_info["trained_model"]
    encoder = config_info["encoder"]
    classes = config_info["classes"] # list of strs
    classes = ", ".join(classes)
    patch_size = config_info["patch_size"]
    log_file = config_info["logs"]


    fig, ax = plt.subplots(3, 1, figsize=(20, 10 * 3))
    sns.lineplot(data=log_file, x="epoch", y="train_loss", ax=ax[0], label="train_loss")
    sns.lineplot(data=log_file, x="epoch", y="val_loss", ax=ax[0], label="val_loss")
    ax[0].set_title(f"Loss, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")

    sns.lineplot(data=log_file, x="epoch", y="train_mean_iou", ax=ax[1], label="train_mean_iou")
    sns.lineplot(data=log_file, x="epoch", y="val_mean_iou", ax=ax[1], label="val_mean_iou")
    ax[1].set_title(f"Mean IoU, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")

    sns.lineplot(data=log_file, x="epoch", y="train_mean_dice", ax=ax[2], label="train_mean_dice")
    sns.lineplot(data=log_file, x="epoch", y="val_mean_dice", ax=ax[2], label="val_mean_dice")
    ax[2].set_title(f"Mean Dice, model: {train_model}, encoder: {encoder}, classes: ({classes}), patch_size: {patch_size}")
    return fig, ax
