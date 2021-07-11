import json
import os
import sys

here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(here, "./lib"))
from dotenv import load_dotenv

load_dotenv()

import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_ID = os.getenv("TELEGRAM_ID")
TELEGRAM_BASE_URL = "https://api.telegram.org/bot{}".format(TELEGRAM_TOKEN)


def lambda_handler(event, context):
    try:
        # context
        data = json.loads(event["body"])
        message = str(data["message"]["text"])
        chat_id = data["message"]["chat"]["id"]
        first_name = data["message"]["chat"]["first_name"]

        response = f"Please say Hi to start the conversation, {first_name}"

        if "Hi" in message:
            response = f"Hi there, {first_name}"

        data = {"text": response.encode("utf8"), "chat_id": chat_id}
        url = TELEGRAM_BASE_URL + "/sendMessage"
        requests.post(url, data)

    except Exception as e:
        print(e)

    return {"statusCode": 200}
