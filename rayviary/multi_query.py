import json
import time

import ray
import requests

test_sizes = [256]

query_count = 0
st = time.time()


@ray.remote(num_cpus=0)
def send_query(text):
    print(f"Sending query '{text}'")
    resp = requests.post("http://localhost:8000/", json={"prompt": text})
    ct = time.time() - st
    print(f"Query sent at {ct}")
    return resp.text


# Let's use Ray to send all queries in parallel
texts = [
    "When was George Washington president?",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
    "Explain the difference between Spark and Ray.",
    "Suggest some fun holiday ideas.",
    "Tell a joke.",
    "What is 2+2?",
    "Explain what is machine learning like I am five years old.",
    "Explain what is artifical intelligence.",
] * 100

for i in test_sizes:
    sublist = texts[:i]
    st = time.time()
    futures = [send_query.remote(text) for text in sublist]
    results = ray.get(futures)
    et = time.time()
    print(f"{i},{et-st}")
    with open(f"results_{i}.json", "w") as f:
        json.dump(results, f)
    time.sleep(10)