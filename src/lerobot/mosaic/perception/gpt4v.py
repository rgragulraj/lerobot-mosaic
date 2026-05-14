"""GPT-4V perception calls for MOSAIC.

Two uses:
  1. classify_shape()       — Phase 0: identify block shape from overhead frame
  2. check_navigate_success() — Phase 2: confirm arm is above the correct slot

Requires: pip install openai
API key:  set OPENAI_API_KEY in environment (or .env loaded by caller)
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

SHAPES = ("circle", "square", "triangle", "star")
ShapeLabel = Literal["circle", "square", "triangle", "star"]

_CLASSIFY_PROMPT = """You are a shape classifier for a robotic sorting task.
The image shows a top-down view of a workspace with one block visible.
The block will be one of these four shapes: circle, square, triangle, star.

Respond ONLY with valid JSON in this exact format:
{"shape": "<shape>", "confidence": <0.0-1.0>}

Where <shape> is one of: circle, square, triangle, star.
Do not include any other text."""

_NAVIGATE_PROMPT = """You are a robotic arm position checker.
The image shows a top-down view of a workspace with a robot arm holding a block.
The workspace has slots for four shapes: circle, square, triangle, star.

Task: Determine whether the robot arm is positioned directly above the {shape} slot.

Respond ONLY with valid JSON in this exact format:
{{"success": true}} or {{"success": false}}

Do not include any other text."""


class GPT4VClient:
    """GPT-4V vision client for shape classification and navigate success checks."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set. Pass api_key= or export the environment variable.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def classify_shape(self, frame: np.ndarray) -> dict:
        """Classify the block shape from a raw overhead camera frame.

        Args:
            frame: Raw BGR frame from the overhead camera. No preprocessing applied.

        Returns:
            {"shape": str, "confidence": float, "latency_ms": int}

        Raises:
            ValueError: If GPT-4V returns an unrecognised shape.
        """
        t0 = time.monotonic()
        image_b64 = _encode_frame(frame)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _CLASSIFY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=64,
            temperature=0,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = response.choices[0].message.content.strip()
        result = _parse_json(raw)

        if result.get("shape") not in SHAPES:
            raise ValueError(f"GPT-4V returned unknown shape: {result.get('shape')!r}. Raw: {raw!r}")

        result["latency_ms"] = latency_ms
        return result

    def check_navigate_success(self, frame: np.ndarray, shape: str) -> dict:
        """Check if the arm is above the correct slot after navigation completes.

        Args:
            frame: Raw BGR frame from the overhead camera. No preprocessing applied.
            shape: Target shape label, e.g. "triangle".

        Returns:
            {"success": bool, "latency_ms": int}

        Raises:
            ValueError: If shape is not one of the four valid labels.
        """
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape: {shape!r}. Must be one of {SHAPES}.")

        t0 = time.monotonic()
        image_b64 = _encode_frame(frame)
        prompt = _NAVIGATE_PROMPT.format(shape=shape)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=32,
            temperature=0,
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = response.choices[0].message.content.strip()
        result = _parse_json(raw)

        if "success" not in result:
            raise ValueError(f"GPT-4V response missing 'success' key. Raw: {raw!r}")

        result["success"] = bool(result["success"])
        result["latency_ms"] = latency_ms
        return result


def _encode_frame(frame: np.ndarray) -> str:
    """Encode a BGR OpenCV frame as a base64 JPEG string for the GPT-4V API."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed — frame may be empty or malformed.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_json(text: str) -> dict:
    """Parse JSON from a GPT response, stripping markdown code fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    return json.loads(text)
