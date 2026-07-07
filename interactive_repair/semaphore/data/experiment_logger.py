"""
SEMAPHORE ExperimentLogger.

Handles all session I/O: save, load, list, export.

Directory layout
----------------
experiments/
  <session_id>/
    session.json        ← full Session document (schema.py format)
    frames/             ← optional frame thumbnails (V1)
    exports/            ← CSV / HDF5 exports (V1)

Every save is atomic: write to .tmp, then rename.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from semaphore.data.schema import (
    load_session, save_session, session_to_dict,
)
from semaphore.data.types import Correction, Session, TrajectoryMetrics
from semaphore.data.managers import CorrectionManager, MetricsManager
from semaphore.data.trajectory_manager import TrajectoryManager

log = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Persists and retrieves session data.

    Usage
    -----
    logger = ExperimentLogger(root_dir="experiments/")
    logger.save(session)
    sessions = logger.list_sessions()
    s = logger.load("some-session-id")
    """

    def __init__(self, root_dir: str | Path = "experiments") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, session: Session) -> Path:
        """Atomically write session to disk. Returns path written."""
        session_dir = self.root / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / "session.json"
        tmp  = session_dir / "session.json.tmp"

        save_session(session, tmp)
        shutil.move(str(tmp), str(dest))
        log.info("Session saved: %s", dest)
        return dest

    def save_snapshot(
        self,
        session: Session,
        tm: TrajectoryManager,
        cm: CorrectionManager,
        mm: MetricsManager,
    ) -> Path:
        """
        Convenience: pull live state from managers into session, then save.
        Call this after every correction or at session end.
        """
        # Update session with live state
        for traj in tm.all_trajectories():
            session.trajectories[traj.point_id] = traj

        session.corrections = cm.all_corrections()

        for metrics in mm.compute_all().values():
            session.metrics[metrics.point_id] = metrics

        return self.save(session)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, session_id: str) -> Session:
        path = self.root / session_id / "session.json"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {path}")
        return load_session(path)

    # ------------------------------------------------------------------
    # List / delete
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        """
        Returns a list of dicts with summary info for each saved session.
        Sorted by creation time descending (newest first).
        """
        summaries = []
        for session_dir in sorted(self.root.iterdir(), reverse=True):
            json_path = session_dir / "session.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path) as f:
                    d = json.load(f)
                summaries.append({
                    "session_id":   d["session_id"],
                    "video_id":     d["video_id"],
                    "tracker":      d["tracker"],
                    "created_at":   d["created_at"],
                    "n_points":     len(d.get("trajectories", {})),
                    "n_corrections": len(d.get("corrections", [])),
                })
            except Exception:
                log.warning("Could not read session at %s", json_path)
        return summaries

    def delete(self, session_id: str) -> None:
        session_dir = self.root / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            log.info("Deleted session: %s", session_id)

    # ------------------------------------------------------------------
    # Export (V0: JSON summary; V1: CSV / HDF5)
    # ------------------------------------------------------------------

    def export_corrections_csv(self, session: Session, dest: Path) -> Path:
        """
        Write all corrections to a flat CSV for external analysis.

        Columns: session_id, point_id, frame_id, old_x, old_y,
                 new_x, new_y, recovery_dist_px, timestamp
        """
        import csv
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "session_id", "point_id", "frame_id",
                "old_x", "old_y", "new_x", "new_y",
                "recovery_dist_px", "status", "timestamp",
            ])
            writer.writeheader()
            for c in session.corrections:
                writer.writerow({
                    "session_id":        session.session_id,
                    "point_id":          c.point_id,
                    "frame_id":          c.frame_id,
                    "old_x":             c.old_position.x,
                    "old_y":             c.old_position.y,
                    "new_x":             c.new_position.x,
                    "new_y":             c.new_position.y,
                    "recovery_dist_px":  round(c.recovery_distance(), 2),
                    "status":            c.status.name,
                    "timestamp":         c.timestamp,
                })
        log.info("Corrections exported to %s", dest)
        return dest

    def export_trajectories_json(self, session: Session, dest: Path) -> Path:
        """Write all trajectories as a standalone JSON file."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "session_id": session.session_id,
            "video_id":   session.video_id,
            "trajectories": {
                pid: {
                    "point_id": t.point_id,
                    "tracker":  t.tracker.value,
                    "points": [
                        {
                            "frame_id":    tp.frame_id,
                            "x":           tp.position.x,
                            "y":           tp.position.y,
                            "confidence":  tp.confidence,
                            "occluded":    tp.occluded,
                            "is_corrected": tp.is_corrected,
                        }
                        for tp in t.points
                    ],
                }
                for pid, t in session.trajectories.items()
            },
        }
        with open(dest, "w") as f:
            json.dump(doc, f, indent=2)
        return dest
