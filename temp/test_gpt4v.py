"""Quick smoke test for gpt4v.py — run from repo root:
python temp/test_gpt4v.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from lerobot.mosaic.perception.gpt4v import GPT4VClient


def main():
    print("Creating GPT4VClient...")
    client = GPT4VClient()
    print("Client created OK — API key loaded successfully\n")

    # Blank frame — won't classify correctly but confirms API connectivity
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("Testing classify_shape()...")
    result = client.classify_shape(dummy_frame)
    print(f"  shape      : {result['shape']}")
    print(f"  confidence : {result['confidence']}")
    print(f"  latency_ms : {result['latency_ms']}\n")

    print("Testing check_navigate_success()...")
    result = client.check_navigate_success(dummy_frame, shape="square")
    print(f"  success    : {result['success']}")
    print(f"  latency_ms : {result['latency_ms']}\n")

    print("All tests passed.")


if __name__ == "__main__":
    main()
