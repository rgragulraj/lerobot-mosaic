"""Data collection logger — records metadata for each demo episode during teleoperation."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

Shape = Literal["square", "rectangle", "star", "cross"]
Quality = Literal["good", "discard"]

SHAPES: list[Shape] = ["square", "rectangle", "star", "cross"]
TARGET_DEMOS_PER_SHAPE = 50


class DataCollectionLogger:
    """
    Logs per-episode metadata during demo recording sessions.

    One session = one invocation of the recording script.
    One episode = one full recorded demo (home → grasp → navigate → insert).

    Writes two files:
      - outputs/collection_logs/session_<id>.json  — this session's episodes
      - outputs/collection_logs/progress.json       — cumulative counts across all sessions
    """

    def __init__(self, log_dir: str | Path = "outputs/collection_logs", operator: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.operator = operator
        self.session_start = datetime.now().isoformat()
        self.episodes: list[dict] = []

        self._progress_path = self.log_dir / "progress.json"
        self._session_path = self.log_dir / f"{self.session_id}.json"

    def log_episode(
        self,
        *,
        shape: Shape,
        episode_id: int,
        keyframe_1_frame: int,
        keyframe_2_frame: int,
        total_frames: int,
        duration_ms: int,
        block_position: str = "",
        quality: Quality = "good",
        notes: str = "",
        dataset_path: str = "",
    ) -> None:
        """Call once per recorded episode, immediately after the episode ends."""
        entry = {
            "episode_id": episode_id,
            "shape": shape,
            "timestamp": datetime.now().isoformat(),
            "keyframe_1_frame": keyframe_1_frame,
            "keyframe_2_frame": keyframe_2_frame,
            "total_frames": total_frames,
            "duration_ms": duration_ms,
            "block_position": block_position,
            "quality": quality,
            "notes": notes,
            "dataset_path": dataset_path,
        }
        self.episodes.append(entry)
        self._flush_session()
        self._update_progress(shape, quality)

    def _flush_session(self) -> None:
        by_shape: dict[str, int] = dict.fromkeys(SHAPES, 0)
        good_by_shape: dict[str, int] = dict.fromkeys(SHAPES, 0)
        for ep in self.episodes:
            by_shape[ep["shape"]] += 1
            if ep["quality"] == "good":
                good_by_shape[ep["shape"]] += 1

        session_doc = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "operator": self.operator,
            "episodes": self.episodes,
            "summary": {
                "total_episodes": len(self.episodes),
                "good_episodes": sum(good_by_shape.values()),
                "by_shape": by_shape,
                "good_by_shape": good_by_shape,
            },
        }
        self._session_path.write_text(json.dumps(session_doc, indent=2))

    def _update_progress(self, shape: Shape, quality: Quality) -> None:
        if self._progress_path.exists():
            progress = json.loads(self._progress_path.read_text())
        else:
            progress = {
                "total_episodes": 0,
                "total_good": 0,
                "by_shape": dict.fromkeys(SHAPES, 0),
                "good_by_shape": dict.fromkeys(SHAPES, 0),
                "target_per_shape": TARGET_DEMOS_PER_SHAPE,
                "sessions": [],
            }

        progress["total_episodes"] += 1
        progress["by_shape"][shape] += 1
        if quality == "good":
            progress["total_good"] += 1
            progress["good_by_shape"][shape] += 1

        if self.session_id not in progress["sessions"]:
            progress["sessions"].append(self.session_id)

        # how far along each shape is toward the target
        progress["completion"] = {
            s: f"{progress['good_by_shape'][s]}/{TARGET_DEMOS_PER_SHAPE}" for s in SHAPES
        }
        progress["all_shapes_complete"] = all(
            progress["good_by_shape"][s] >= TARGET_DEMOS_PER_SHAPE for s in SHAPES
        )

        self._progress_path.write_text(json.dumps(progress, indent=2))

    def print_progress(self) -> None:
        """Print current collection progress to stdout."""
        if not self._progress_path.exists():
            print("No progress data yet.")
            return
        progress = json.loads(self._progress_path.read_text())
        print("\n--- Data Collection Progress ---")
        for shape in SHAPES:
            good = progress["good_by_shape"][shape]
            pct = int(good / TARGET_DEMOS_PER_SHAPE * 100)
            bar = "#" * (pct // 5) + "." * (20 - pct // 5)
            print(f"  {shape:<10} [{bar}] {good:>3}/{TARGET_DEMOS_PER_SHAPE}  ({pct}%)")
        total_good = progress["total_good"]
        total_target = TARGET_DEMOS_PER_SHAPE * len(SHAPES)
        print(f"\n  Overall: {total_good}/{total_target} good demos")
        if progress.get("all_shapes_complete"):
            print("  ✓ All shapes complete — ready to split and train.")
        print()
