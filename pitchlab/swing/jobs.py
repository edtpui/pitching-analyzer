"""Swing analyses on disk, run by the app's shared background worker.

data/swings/<id>/ holds meta.json (status, options), the upload, pose.npz, result.json, overlay.mp4 and
keyframes/, like a pitching-mechanics analysis.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from ..jobs import Jobs, now
from .pipeline import Options, options_dict, read_json, run


def summary_of(result: dict) -> dict:
    """The few numbers the swings list shows, recomputed whenever result.json changes."""
    return {"bats": result["athlete"]["bats"],
            "fps": round(result["video"]["fps_effective"]),
            "camera_view": result["quality"]["camera_view"],
            "warnings": len(result["quality"]["warnings"]),
            "predictions": {p["target"]: round(p["value"], 1) for p in result.get("predictions", []) if p["reportable"]}}


class SwingJobs(Jobs):
    def create(self, filename: str, suffix: str, opts: Options) -> tuple[str, Path]:   # type: ignore[override]
        swing_id = uuid.uuid4().hex[:12]
        d = self.root / swing_id
        d.mkdir(parents=True)
        self._write_meta(swing_id, id=swing_id, kind="swing", created_utc=now(), filename=filename[:200],
                         input_file=f"input{suffix}", options=options_dict(opts), status="uploading",
                         progress=0.0, stage="Uploading")
        return swing_id, d

    def update_summary(self, swing_id: str, result: dict) -> None:
        self._write_meta(swing_id, summary=summary_of(result))

    def _run(self, swing_id: str) -> None:
        d = self.root / swing_id
        m = read_json(d / "meta.json")
        self._write_meta(swing_id, status="running", started_utc=now(), stage="Starting")
        t0 = time.time()
        try:
            opts = Options.from_dict(m["options"])
            result = run(d, opts, d / m["input_file"], lambda f, s: self._progress(swing_id, f, s))
            self._write_meta(swing_id, status="done", progress=1.0, stage="Done", finished_utc=now(),
                             seconds=round(time.time() - t0, 1), summary=summary_of(result))
        except Exception as exc:  # noqa: BLE001 - _fail tells user-facing failures from crashes
            self._fail(swing_id, exc)
        finally:
            with self._lock:
                self._live.pop(swing_id, None)
