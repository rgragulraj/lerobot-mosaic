"""Quick test: capture one frame from the overhead camera and run GPT-4V shape classification.

Usage:
    uv run python scripts/test_gpt4v.py
    uv run python scripts/test_gpt4v.py --cam /dev/video5 --save frame.jpg
"""

import argparse
import sys
from contextlib import suppress
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from lerobot.mosaic.perception.gpt4v import GPT4VClient

OVERHEAD_CAM = "/dev/video5"


def main():
    parser = argparse.ArgumentParser(description="Test GPT-4V on overhead camera")
    parser.add_argument("--cam", default=OVERHEAD_CAM, help="Camera device path or index")
    parser.add_argument("--save", default="", help="Save captured frame to this path (optional)")
    args = parser.parse_args()

    cam_arg = args.cam
    with suppress(ValueError):
        cam_arg = int(cam_arg)

    print(f"Opening camera: {args.cam}")
    cap = cv2.VideoCapture(cam_arg)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.cam}")
        sys.exit(1)

    # Warm up — discard the first few frames (some cameras need a moment)
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("ERROR: failed to capture frame.")
        sys.exit(1)

    print(f"Frame captured: {frame.shape[1]}x{frame.shape[0]}")

    print("Sending to GPT-4V...")
    client = GPT4VClient()
    result = client.classify_shape(frame)

    print()
    print(f"  Shape      : {result['shape']}")
    print(f"  Confidence : {result['confidence']:.2f}")
    print(f"  Latency    : {result['latency_ms']} ms")

    annotated = frame.copy()
    bbox = result.get("bbox")
    if bbox and len(bbox) == 4:
        h, w = annotated.shape[:2]
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{result['shape']} {result['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    save_path = Path(args.save) if args.save else Path("outputs/collection_logs/gpt4v_test.jpg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), annotated)
    print(f"  Annotated frame saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
