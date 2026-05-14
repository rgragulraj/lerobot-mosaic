"""
Interactive CLAHE preview for two cameras (overhead + gripper).

Shows raw vs CLAHE side-by-side for each camera with live trackbars
so you can tune clip_limit and tile_grid_size before committing to
adding CLAHE to the processing pipeline.

Usage:
    python scripts/preview_clahe.py
    python scripts/preview_clahe.py --overhead 0 --gripper 2
    python scripts/preview_clahe.py --overhead /dev/video0 --gripper /dev/video2
    python scripts/preview_clahe.py --width 640 --height 480 --fps 30

Controls:
    q          quit
    s          save a screenshot of the current view
    CLIP       trackbar — CLAHE clip limit * 10  (so 20 = clip_limit 2.0)
    TILE       trackbar — CLAHE tile grid size
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW = "CLAHE Preview  [q=quit  s=save]"
TRACKBAR_CLIP = "Clip x10"
TRACKBAR_TILE = "Tile grid"

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.55
LABEL_THICKNESS = 1
LABEL_COLOR = (255, 255, 255)
LABEL_SHADOW = (0, 0, 0)


def apply_clahe(bgr: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    (tw, th), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    x, y = 8, th + 8
    cv2.putText(img, text, (x + 1, y + 1), LABEL_FONT, LABEL_SCALE, LABEL_SHADOW, LABEL_THICKNESS + 1)
    cv2.putText(img, text, (x, y), LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS)
    return img


def open_camera(index_or_path: str | int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index_or_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {index_or_path}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def read_frame(cap: cv2.VideoCapture, name: str) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"[warn] failed to read from {name}")
        return None
    return frame


def build_panel(
    overhead_raw: np.ndarray | None,
    gripper_raw: np.ndarray | None,
    clip_limit: float,
    tile_grid_size: int,
    fps: float,
    display_w: int,
    display_h: int,
) -> np.ndarray:
    placeholder = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No signal", (10, display_h // 2), LABEL_FONT, 1.0, (80, 80, 80), 2)

    def make_pair(raw: np.ndarray | None, cam_name: str):
        if raw is None:
            return np.hstack([placeholder, placeholder])
        resized = cv2.resize(raw, (display_w, display_h))
        processed = apply_clahe(resized, clip_limit, tile_grid_size)
        left = label(resized, f"{cam_name} — raw")
        right = label(
            processed, f"{cam_name} — CLAHE  clip={clip_limit:.1f}  tile={tile_grid_size}x{tile_grid_size}"
        )
        return np.hstack([left, right])

    top = make_pair(overhead_raw, "overhead")
    bot = make_pair(gripper_raw, "gripper")

    panel = np.vstack([top, bot])

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(panel, fps_text, (panel.shape[1] - 110, 24), LABEL_FONT, 0.6, LABEL_SHADOW, 2)
    cv2.putText(panel, fps_text, (panel.shape[1] - 111, 23), LABEL_FONT, 0.6, (0, 255, 120), 1)

    return panel


def main():
    parser = argparse.ArgumentParser(description="CLAHE live preview — overhead + gripper cameras")
    parser.add_argument(
        "--overhead", default=0, help="Overhead camera index or /dev/videoN path (default: 0)"
    )
    parser.add_argument("--gripper", default=1, help="Gripper camera index or /dev/videoN path (default: 1)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--display-width", type=int, default=480, help="Width of each camera panel (default: 480)"
    )
    parser.add_argument(
        "--display-height", type=int, default=360, help="Height of each camera panel (default: 360)"
    )
    parser.add_argument("--clip", type=float, default=2.0, help="Initial CLAHE clip limit (default: 2.0)")
    parser.add_argument("--tile", type=int, default=8, help="Initial CLAHE tile grid size (default: 8)")
    parser.add_argument("--save-dir", default=".", help="Directory for saved screenshots")
    args = parser.parse_args()

    # Parse camera args — allow int or string paths
    def parse_cam_arg(val):
        try:
            return int(val)
        except ValueError:
            return val

    overhead_idx = parse_cam_arg(args.overhead)
    gripper_idx = parse_cam_arg(args.gripper)

    print(f"Opening overhead camera: {overhead_idx}")
    print(f"Opening gripper  camera: {gripper_idx}")

    try:
        cap_overhead = open_camera(overhead_idx, args.width, args.height, args.fps)
    except RuntimeError as e:
        print(f"[error] {e}")
        cap_overhead = None

    try:
        cap_gripper = open_camera(gripper_idx, args.width, args.height, args.fps)
    except RuntimeError as e:
        print(f"[error] {e}")
        cap_gripper = None

    if cap_overhead is None and cap_gripper is None:
        print("No cameras available. Exiting.")
        return

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    panel_w = args.display_width * 2  # raw + clahe side by side
    panel_h = args.display_height * 2  # overhead + gripper stacked
    cv2.resizeWindow(WINDOW, panel_w, panel_h)

    # Trackbars: clip stored as integer * 10 so we get one decimal place
    clip_init = max(1, int(args.clip * 10))
    cv2.createTrackbar(TRACKBAR_CLIP, WINDOW, clip_init, 200, lambda _: None)
    cv2.createTrackbar(TRACKBAR_TILE, WINDOW, max(2, args.tile), 32, lambda _: None)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    frame_times: list[float] = []
    fps_display = 0.0

    print("Preview running — press 'q' to quit, 's' to save screenshot")

    while True:
        t0 = time.perf_counter()

        clip_raw = cv2.getTrackbarPos(TRACKBAR_CLIP, WINDOW)
        tile_raw = cv2.getTrackbarPos(TRACKBAR_TILE, WINDOW)
        clip_limit = max(0.1, clip_raw / 10.0)
        tile_grid_size = max(2, tile_raw)

        overhead_frame = read_frame(cap_overhead, "overhead") if cap_overhead else None
        gripper_frame = read_frame(cap_gripper, "gripper") if cap_gripper else None

        panel = build_panel(
            overhead_frame,
            gripper_frame,
            clip_limit,
            tile_grid_size,
            fps_display,
            args.display_width,
            args.display_height,
        )

        cv2.imshow(WINDOW, panel)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"clahe_preview_{ts}.png"
            cv2.imwrite(str(path), panel)
            print(f"Screenshot saved: {path}")

        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps_display = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    if cap_overhead:
        cap_overhead.release()
    if cap_gripper:
        cap_gripper.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
