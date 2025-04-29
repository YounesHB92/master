import os

from src.utils import load_env_variables

_ = load_env_variables()

print(os.getenv("CEHCKPOINTS_DIR"))