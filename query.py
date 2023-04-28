import argparse
import json
import time

import requests


def send_query(text, host):
    print(f"Sending query '{text}'")
    resp = requests.post(host, json={"prompt": text})
    return resp.text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        type=str,
        help="Prompt",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="Host to query",
    )
    args = parser.parse_args()

    print(send_query(args.prompt, args.host))
