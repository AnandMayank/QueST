"""One-off headed napari demo: load a real SAPIEN sequence, track two points
via SEMAPHORE's bridged CoTracker backend, screenshot the result.

Run with the movement venv (has napari), with SEMAPHORE on the path:
    DISPLAY=:1 ~/movement/.venv/bin/python \
        -m <path-to-this-repo>/downstream_causal/napari_demo.py
"""

import os
from pathlib import Path
import sys

# Appended, not prepended: <path-to-this-repo> contains an unrelated
# "sparse" subdirectory that otherwise shadows the real PyData `sparse`
# package dask depends on.
sys.path.append(str(Path(__file__).resolve().parent.parent / "interactive_repair"))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import napari

# Creating the Viewer first (as in the earlier successful headless check)
# triggers napari's lazy submodule loading (napari.layers -> dask -> sparse)
# in a working order; importing semaphore.ui.napari_widget (which pulls in
# napari.layers directly) before a Viewer exists hits a dask/sparse version
# mismatch in this venv.
_viewer_bootstrap = napari.Viewer(show=False)
_viewer_bootstrap.close()

from semaphore.ui.napari_widget import SemaphoreWidget
from semaphore.data.types import Point2D, Session, TrackerKind
from semaphore.data.trajectory_manager import TrajectoryManager
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.schema import new_session_id
from semaphore.backends.bridge import BridgedCoTrackerBackend

from downstream_causal.data import SyntheticSequence

SEQ_DIR = os.environ.get("QUEST_DEMO_SEQUENCE", "<path-to-quest_partnet_subset>/manipulation_4/45189/take_08")


def main():
    seq = SyntheticSequence.load(SEQ_DIR)
    video = seq.load_frames()
    print(f"loaded {video.shape} video, {len(seq.part_ids)} parts")

    viewer = napari.Viewer(show=True, title="SEMAPHORE Phase 2 demo")
    widget = SemaphoreWidget(viewer)
    widget._add_video_layer(video, fps=15.0)

    backend = BridgedCoTrackerBackend(tracker_name="cotracker3", device="cuda")
    backend.load_video(SEQ_DIR)
    tm = TrajectoryManager(backend, video, video_id="demo")
    cm = CorrectionManager(tm, video_id="demo")
    mm = MetricsManager(tm, cm)
    widget._traj_manager, widget._corr_manager, widget._metrics_manager = tm, cm, mm
    widget._session = Session(
        session_id=new_session_id(), video_id="demo", video_path=SEQ_DIR,
        tracker=TrackerKind.COTRACKER,
    )

    # Two real query points on two different parts -- this is the exact
    # switch-prone setup E1/E2/E4 study.
    rng_points = [seq.gt_frames[0].centers[p] for p in seq.part_ids[:2]]
    print("tracking points via bridge (real CoTracker3 in vidbot conda env)...")
    tm.add_points_batch(
        [Point2D(float(c[0]), float(c[1])) for c in rng_points], start_frame=0
    )
    widget._refresh_tracks_layer()
    widget._refresh_metrics()
    widget._set_status(f"Demo: {len(rng_points)} points tracked with CoTracker3 (bridged).")
    print("tracking complete, metrics:\n", widget.metrics_label.value)

    viewer.window.qt_viewer.canvas.native.repaint()
    import time
    time.sleep(1.0)  # let the compositor draw before screenshot

    out_path = str(Path(__file__).resolve().parent / "results" / "napari_demo_screenshot.png")
    viewer.screenshot(path=out_path, canvas_only=False)
    print(f"screenshot saved: {out_path}")

    viewer.close()


if __name__ == "__main__":
    main()
