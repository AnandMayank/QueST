"""
scripts/run_headless.py

Run SEMAPHORE in headless mode (no Napari).
Useful for benchmarking, automated testing, and CI.

Usage
-----
python scripts/run_headless.py \
    --video path/to/video.mp4 \
    --clicks "0,320,240" "0,100,100" \
    --tracker cotracker \
    --device cpu \
    --output experiments/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Make semaphore importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from semaphore.backends.cotracker import CoTrackerBackend
from semaphore.backends.tapir import TapirBackend
from semaphore.data.experiment_logger import ExperimentLogger
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.schema import new_session_id
from semaphore.data.trajectory_manager import TrajectoryManager
from semaphore.data.types import Point2D, Session, TrackerKind
from semaphore.utils.video_loader import VideoLoader


BACKENDS = {
    "cotracker": CoTrackerBackend,
    "tapir":     TapirBackend,
}


def parse_click(s: str) -> tuple[int, Point2D]:
    """Parse "frame,x,y" string → (frame_id, Point2D)."""
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Expected 'frame,x,y' but got: {s!r}")
    return int(parts[0]), Point2D(float(parts[1]), float(parts[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="SEMAPHORE headless runner")
    parser.add_argument("--video",   required=True, help="Path to input video")
    parser.add_argument("--clicks",  nargs="+", default=[],
                        help='Click specs as "frame,x,y" strings')
    parser.add_argument("--tracker", choices=list(BACKENDS), default="cotracker")
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--output",  default="experiments",
                        help="Root directory for experiment output")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Limit video to N frames (for quick tests)")
    args = parser.parse_args()

    # -- Load video --
    loader = VideoLoader()
    frames = loader.load(args.video, max_frames=args.max_frames)

    # -- Build backend --
    BackendClass = BACKENDS[args.tracker]
    backend = BackendClass(device=args.device)
    backend.load_video(args.video)

    # -- Session --
    session = Session(
        session_id=new_session_id(),
        video_id=loader.video_id,
        video_path=args.video,
        tracker=TrackerKind(args.tracker),
    )

    tm = TrajectoryManager(backend, frames, video_id=loader.video_id)
    cm = CorrectionManager(tm, video_id=loader.video_id)
    mm = MetricsManager(tm, cm)
    logger = ExperimentLogger(root_dir=args.output)

    # -- Track each click --
    for click_str in args.clicks:
        frame_id, position = parse_click(click_str)
        logging.info("Tracking click: frame=%d  pos=(%.1f, %.1f)", frame_id, position.x, position.y)
        traj = tm.add_point(position, start_frame=frame_id)
        logging.info("  → %d trajectory points", traj.length())

    # -- Print metrics --
    all_metrics = mm.compute_all()
    for pid, m in all_metrics.items():
        print(
            f"Point {pid[:8]}  corrections={m.total_corrections}  "
            f"effort={m.human_effort_score:.3f}  "
            f"mean_confidence={m.mean_confidence:.3f}"
        )

    # -- Save --
    saved_path = logger.save_snapshot(session, tm, cm, mm)
    print(f"\nSession saved: {saved_path}")


if __name__ == "__main__":
    main()
