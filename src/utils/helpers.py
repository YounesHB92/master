from tqdm import tqdm
import sys
from dotenv import load_dotenv
import yaml

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

    encoder_name = parts[encoder_index+1:epoch_index]
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
        dotenv_path = "/content/master/.env." + environment
    else:
        dotenv_path = ".env." + environment
    load_dotenv(dotenv_path=dotenv_path)
    return environment

def find_configs():
    env = find_env()
    if env == "colab":
        configs_file = "/content/master/colab_configs.yaml"
    else:
        configs_file = "local_configs.yaml"
    with open(configs_file, "r") as file:
        configs = yaml.safe_load(file)
    return configs