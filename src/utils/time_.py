import pytz
from datetime import datetime


def add_time(name):
    time_zone = pytz.timezone("Australia/Brisbane")
    name_time_added = f"{name}_{datetime.now(time_zone).strftime('%Y-%m-%d_%H-%M-%S')}"
    return name_time_added