import sys

import numpy as np
from tqdm import tqdm
from twilio.rest import Client


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


def inspect_object(obj):
    print(f"Inspecting object: {obj.__class__.__name__}")
    for attr, value in vars(obj).items():
        print(f"{attr}: {value}")


def tqdm_print(*args, **kwargs):
    return tqdm(*args, file=sys.stdout, **kwargs)


def print_indented(text, level=1):
    indent = "\t" * level
    print(indent + text)


def send_sms(message, sid, token, from_, to):
    account_sid = sid
    auth_token = token
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_=from_,
        to=to,
        body=message)
    print(message.sid)


def remove_correlated_features(df, threshold=0.8):
    # Compute correlation matrix (only numeric columns)
    corr_matrix = df.select_dtypes(include=[float, int]).corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop correlated features
    df_cleaned = df.drop(columns=to_drop)

    print(f"Removed {len(to_drop)} features due to high correlation.")
    return df_cleaned
