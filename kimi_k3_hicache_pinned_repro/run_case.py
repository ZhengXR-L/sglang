import json
import os
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen


base_url = sys.argv[1]
count = int(os.environ["REQUEST_COUNT"])
prefix_tokens = int(os.environ["PREFIX_TOKENS"])
timeout = float(os.environ["REQUEST_TIMEOUT"])
if not 2 <= count <= 100 or prefix_tokens < 1 or timeout <= 0:
    raise ValueError("Require 2–100 requests, a positive prefix length and timeout")

prefix = [100 + index % 16 for index in range(prefix_tokens)]
for index in range(count):
    payload = {
        "input_ids": [*prefix, 200 + index],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 8,
            "ignore_eos": True,
        },
    }
    request = Request(
        base_url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"request {index + 1}/{count}, shared_prefix_tokens={prefix_tokens}")
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        print(f"HTTP {error.code}: {error.read(4096).decode(errors='replace')}")
        raise

    metadata = result.get("meta_info", {})
    print(json.dumps(metadata, ensure_ascii=False))
    if metadata.get("completion_tokens") != 8:
        raise RuntimeError("Expected 8 completion tokens")
