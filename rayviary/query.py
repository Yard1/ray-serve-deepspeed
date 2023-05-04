import argparse
from pprint import pprint

import requests


def send_query(text, host):
    print(f"Sending query '{text}'")
    resp = requests.post(host, json={"prompt": text})
    try:
        return resp.json()
    except Exception:
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

    pprint(send_query(args.prompt, args.host))

text = "Can you recommend some books to read?"
resp = requests.post(
    "https://aviary-v0-jrvwy.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata-staging.com/query/CarperAI--stable-vicuna-13b-delta",
    headers={"Authorization": "Bearer LBGTsD3WEJiapmMnEOnLBt1IfneZIWoeeOGMjgAfpiI"},
    json={"prompt": text},
)
