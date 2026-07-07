"""
SEMAPHORE Napari plugin (magicgui widget approach).

Registers a dock widget in Napari that drives the full SEMAPHORE pipeline:
  1. User loads video via file chooser.
  2. User selects tracker backend.
  3. User clicks on the video to initialise tracking.
  4. Trajectories are displayed as Points + Tracks layers.
  5. User clicks in correction mode to fix drift.
  6. Metrics panel updates live.

Installation
------------
Register with Napari by adding this to napari.yaml:

    name: semaphore
    contributions:
      widgets:
        - name: SEMAPHORE
          command: semaphore.ui.napari_widget:semaphore_widget
          display_name: SEMAPHORE — Point Tracker

Or call SemaphoreWidget(napari_viewer) directly from a script.
"""

from __future__ import annotations

import logging
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Try to import Napari — degrade gracefully if not installed
# ---------------------------------------------------------------------------
try:
    import napari
    from napari.layers import Image, Points, Tracks
    from napari.types import FullLayerData
    HAS_NAPARI = True
except ImportError:
    HAS_NAPARI = False
    log.warning("Napari not found — NapariUI is unavailable.")

try:
    from magicgui import magicgui
    from magicgui.widgets import (
        ComboBox, Container, FileEdit, Label, PushButton, SpinBox,
    )
    HAS_MAGICGUI = True
except ImportError:
    HAS_MAGICGUI = False


from semaphore.backends.cotracker import CoTrackerBackend
from semaphore.backends.tapir import TapirBackend
from semaphore.backends.bridge import BridgedCoTrackerBackend
from semaphore.data.experiment_logger import ExperimentLogger
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.schema import new_session_id
from semaphore.data.trajectory_manager import TrajectoryManager
from semaphore.data.types import Point2D, Session, TrackerKind
from semaphore.utils.video_loader import VideoLoader

TRACKER_OPTIONS = {
    "CoTracker": CoTrackerBackend,
    "TAPIR": TapirBackend,
    # Runs in the vidbot conda env via subprocess -- use this when the GUI
    # process itself has no torch (e.g. the napari-only venv this ships
    # with), see semaphore/backends/bridge.py.
    "CoTracker (bridged)": BridgedCoTrackerBackend,
}

# Layer names (constants so we can find them reliably)
LAYER_VIDEO    = "SEMAPHORE: video"
LAYER_TRACKS   = "SEMAPHORE: tracks"
LAYER_POINTS   = "SEMAPHORE: points"
LAYER_CORRECT  = "SEMAPHORE: corrections"


class InteractionMode(str, Enum):
    TRACK   = "track"       # next click initialises a new point
    CORRECT = "correct"     # next click corrects the nearest point


class SemaphoreWidget:
    """
    Main controller / dock widget for Napari.

    Instantiate via semaphore_widget() factory below (magicgui integration),
    or directly: SemaphoreWidget(viewer) from a script.
    """

    def __init__(self, viewer: "napari.Viewer") -> None:
        self.viewer = viewer
        self.mode   = InteractionMode.TRACK
        self.logger = ExperimentLogger(root_dir="experiments")

        # Runtime state (set after video load)
        self._video_loader:      Optional[VideoLoader]        = None
        self._session:           Optional[Session]            = None
        self._traj_manager:      Optional[TrajectoryManager]  = None
        self._corr_manager:      Optional[CorrectionManager]  = None
        self._metrics_manager:   Optional[MetricsManager]     = None
        self._selected_point_id: Optional[str]                = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        if not HAS_MAGICGUI:
            log.warning("magicgui not available; UI not constructed.")
            return

        # -- File chooser --
        self.file_edit = FileEdit(
            label="Video file",
            mode="r",
            filter="Video files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )

        # -- Tracker selector --
        self.tracker_combo = ComboBox(
            label="Tracker",
            choices=list(TRACKER_OPTIONS.keys()),
            value="CoTracker",
        )

        # -- Device --
        self.device_combo = ComboBox(
            label="Device",
            choices=["cpu", "cuda", "mps"],
            value="cpu",
        )

        # -- Load button --
        self.load_btn = PushButton(label="Load video")
        self.load_btn.clicked.connect(self._on_load_video)

        # -- Mode toggle --
        self.mode_btn = PushButton(label="Mode: TRACK  (click to switch)")
        self.mode_btn.clicked.connect(self._on_toggle_mode)

        # -- Undo --
        self.undo_btn = PushButton(label="Undo last correction")
        self.undo_btn.clicked.connect(self._on_undo)

        # -- Save --
        self.save_btn = PushButton(label="Save session")
        self.save_btn.clicked.connect(self._on_save)

        # -- Status label --
        self.status_label = Label(value="Load a video to begin.")
        self.metrics_label = Label(value="")

        self.container = Container(widgets=[
            self.file_edit,
            self.tracker_combo,
            self.device_combo,
            self.load_btn,
            self.mode_btn,
            self.undo_btn,
            self.save_btn,
            self.status_label,
            self.metrics_label,
        ])

        self.viewer.window.add_dock_widget(
            self.container, area="right", name="SEMAPHORE"
        )

    # ------------------------------------------------------------------
    # Load video
    # ------------------------------------------------------------------

    def _on_load_video(self) -> None:
        path = str(self.file_edit.value)
        if not path or path == ".":
            self._set_status("Please select a video file.")
            return

        self._set_status(f"Loading {Path(path).name} …")
        loader = VideoLoader()
        frames = loader.load(path)

        # Choose backend
        tracker_name = self.tracker_combo.value
        device       = self.device_combo.value
        BackendClass = TRACKER_OPTIONS[tracker_name]
        backend = BackendClass(device=device)
        backend.load_video(path)

        # Build session and managers
        session = Session(
            session_id=new_session_id(),
            video_id=loader.video_id,
            video_path=path,
            tracker=TrackerKind(tracker_name.lower()),
        )

        tm = TrajectoryManager(backend, frames, video_id=loader.video_id)
        cm = CorrectionManager(tm, video_id=loader.video_id)
        mm = MetricsManager(tm, cm)

        self._video_loader    = loader
        self._session         = session
        self._traj_manager    = tm
        self._corr_manager    = cm
        self._metrics_manager = mm

        # Add video layer
        self._add_video_layer(frames, loader.fps)

        self._set_status(
            f"Loaded {loader.n_frames} frames | {tracker_name} on {device}\n"
            f"Mode: TRACK — click a point on the video."
        )

    # ------------------------------------------------------------------
    # Click handling (hooked into napari Points layer)
    # ------------------------------------------------------------------

    def _add_video_layer(self, frames: np.ndarray, fps: float) -> None:
        """Add the video as an Image layer with the frame axis as time."""
        # Remove old layer if exists
        for lname in [LAYER_VIDEO, LAYER_TRACKS, LAYER_POINTS, LAYER_CORRECT]:
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)

        self.viewer.add_image(
            frames,
            name=LAYER_VIDEO,
            rgb=True,
        )

        # Empty Points layer — user clicks here to place points
        click_layer = self.viewer.add_points(
            np.empty((0, 3)),   # (N, [t, y, x])
            name=LAYER_POINTS,
            size=10,
            face_color="lime",
            border_color="white",
            ndim=3,
        )
        click_layer.mode = "add"
        click_layer.editable = True
        self.viewer.layers.selection.active = click_layer
        click_layer.events.data.connect(self._on_point_added)

    def _on_point_added(self, event) -> None:
        """Fired when user clicks to add a point in the Points layer."""
        if self._traj_manager is None:
            return

        pts_layer = self.viewer.layers[LAYER_POINTS]
        if len(pts_layer.data) == 0:
            return

        # Most recently added point (last row)
        last = pts_layer.data[-1]          # [t, y, x]
        frame_id = int(last[0])
        position = Point2D(x=float(last[2]), y=float(last[1]))

        if self.mode == InteractionMode.TRACK:
            self._handle_new_point(frame_id, position)
        elif self.mode == InteractionMode.CORRECT:
            self._handle_correction(frame_id, position)

        # Keep the seed point visible so the user can see what was added.
        # The trajectory itself is displayed via the Tracks layer.

    def _handle_new_point(self, frame_id: int, position: Point2D) -> None:
        self._set_status(f"Tracking from frame {frame_id} …")
        try:
            traj = self._traj_manager.add_point(
                initial_position=position,
                start_frame=frame_id,
            )
            self._selected_point_id = traj.point_id
            self._refresh_tracks_layer()
            self._refresh_metrics()
            self._set_status(
                f"Tracking complete: {len(traj.points)} frames\n"
                f"Point ID: {traj.point_id[:8]}…"
            )
        except Exception as exc:
            self._set_status(f"Tracking failed: {exc}")
            log.exception("Tracking error")

    def _handle_correction(self, frame_id: int, position: Point2D) -> None:
        point_id = self._selected_point_id

        # If nothing is selected, or the selected point is not close to the
        # click, pick the nearest visible track at this frame.
        if point_id is None:
            point_id = self._traj_manager.nearest_point_id_at_frame(frame_id, position)
        else:
            selected_pos = self._traj_manager.get_position_at_frame(point_id, frame_id)
            if selected_pos is None or selected_pos.distance_to(position) > 40.0:
                point_id = self._traj_manager.nearest_point_id_at_frame(frame_id, position)

        if point_id is None:
            self._set_status(
                "No nearby track found at this frame. Switch to TRACK mode and click a point first."
            )
            return

        self._selected_point_id = point_id

        self._set_status(f"Applying correction at frame {frame_id} …")
        try:
            correction = self._corr_manager.apply(
                point_id=point_id,
                frame_id=frame_id,
                corrected_position=position,
            )
            self._refresh_tracks_layer()
            self._refresh_metrics()
            self._set_status(
                f"Correction applied to {point_id[:8]}…: drift={correction.recovery_distance():.1f}px\n"
                f"Total corrections: {self._corr_manager.correction_count()}"
            )
        except Exception as exc:
            self._set_status(f"Correction failed: {exc}")
            log.exception("Correction error")

    # ------------------------------------------------------------------
    # Layer refresh
    # ------------------------------------------------------------------

    def _refresh_tracks_layer(self) -> None:
        if self._traj_manager is None:
            return

        track_data = self._traj_manager.tracks_for_napari()  # (N, 4): id,t,y,x

        if LAYER_TRACKS in self.viewer.layers:
            self.viewer.layers.remove(LAYER_TRACKS)

        if len(track_data) > 0:
            self.viewer.add_tracks(
                track_data,
                name=LAYER_TRACKS,
                tail_length=50,
                head_length=0,
                tail_width=3,
                colormap="hsv",
            )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _on_toggle_mode(self) -> None:
        if self.mode == InteractionMode.TRACK:
            self.mode = InteractionMode.CORRECT
            self.mode_btn.label = "Mode: CORRECT  (click to switch)"
            self._set_status(
                "Mode switched to CORRECT — click near an existing track to correct it."
            )
        else:
            self.mode = InteractionMode.TRACK
            self.mode_btn.label = "Mode: TRACK  (click to switch)"
            self._set_status(
                "Mode switched to TRACK — click on the video to add a new point."
            )

    def _on_undo(self) -> None:
        if self._corr_manager is None:
            return
        if not self._corr_manager.can_undo():
            self._set_status("Nothing to undo.")
            return
        self._corr_manager.undo()
        self._refresh_tracks_layer()
        self._refresh_metrics()
        self._set_status("Last correction undone.")

    def _on_save(self) -> None:
        if self._session is None:
            self._set_status("Nothing to save yet.")
            return
        path = self.logger.save_snapshot(
            self._session,
            self._traj_manager,
            self._corr_manager,
            self._metrics_manager,
        )
        self._set_status(f"Session saved:\n{path}")

    # ------------------------------------------------------------------
    # Metrics display
    # ------------------------------------------------------------------

    def _refresh_metrics(self) -> None:
        if self._metrics_manager is None:
            return
        all_m = self._metrics_manager.compute_all()
        if not all_m:
            return
        lines = ["── Metrics ──"]
        for pid, m in all_m.items():
            traj = self._traj_manager.get_trajectory(pid)
            n_frames = traj.length() if traj is not None else 0
            lines.append(
                f"Point {pid[:6]}…  "
                f"frames={n_frames}  "
                f"corrections={m.total_corrections}  "
                f"rate={m.correction_rate:.2f}  "
                f"effort={m.human_effort_score:.2f}  "
                f"conf={m.mean_confidence:.2f}  "
                f"occ={m.occlusion_rate:.2f}"
            )
        self.metrics_label.value = "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        log.info(msg)
        self.status_label.value = msg


# ---------------------------------------------------------------------------
# Factory function registered with Napari
# ---------------------------------------------------------------------------

def semaphore_widget(viewer: "napari.Viewer") -> SemaphoreWidget:
    """Entry point registered in napari.yaml."""
    return SemaphoreWidget(viewer)
