import os

from src.utils import load_env_variables
import src.utils as utils

_ = load_env_variables()

sid = os.environ['TWILIO_SID']
token = os.environ['TWILIO_TOKEN']

message = """
Hey buddy, how's it going today.
"""

response = utils.send_sms(
    sid=sid,
    token=token,
    message=message,
)

print(response)