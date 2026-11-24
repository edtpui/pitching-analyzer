"""Ball-flight analyses on disk and in the shared background worker.

data/flights/<id>/ holds meta.json, the upload, background.png/.jpg (the people-removed frame used for
marking the field and for detection), calibration.json, track.npz, result.json, tracer.mp4 and seams.jpg.
Status runs uploading -> queued -> running (prepare) -> calibrate (waiting for the marks) -> queued ->
running (analyze) -> done | failed; a new calibration or a re-run goes back through queued.
"""
from __future__ import annotations

import time
import uuid

from ..jobs import Jobs, now
from ..pipeline import read_json
from .pipeline import FlightOptions, analyze, options_dict, prepare


def summary_of(result: dict) -> dict:
    p, spin = result["pitch"], result.get("spin") or {}
    return {"release_speed_mph": round(p["release_speed_mph"], 1),
            "spin_rpm": round(spin["rpm"]) if spin.get("ok") else None,
            "tilt": spin.get("tilt_clock") if spin.get("ok") else result.get("movement_spin", {}).get("tilt_clock"),
            "ivb_in": None if p.get("ivb_in") is None else round(p["ivb_in"], 1),
            "hb_arm_side_in": None if p.get("hb_arm_side_in") is None else round(p["hb_arm_side_in"], 1),
            "strike": p.get("strike"), "fps": round(result["video"]["fps_effective"]),
            "release_seen": result["release"]["observed"], "warnings": len(result["quality"]["warnings"])}


class FlightJobs(Jobs):
    def create(self, filename: str, suffix: str, opts: FlightOptions) -> tuple[str, object]:   # type: ignore[override]
        flight_id = uuid.uuid4().hex[:12]
        d = self.root / flight_id
        d.mkdir(parents=True)
        self._write_meta(flight_id, id=flight_id, kind="flight", created_utc=now(), filename=filename[:200],
                         input_file=f"input{suffix}", options=options_dict(opts), status="uploading", phase="prepare",
                         progress=0.0, stage="Uploading")
        return flight_id, d

    def submit(self, flight_id: str, phase: str = "prepare") -> None:     # type: ignore[override]
        self._write_meta(flight_id, status="queued", phase=phase, stage="Waiting in queue", progress=0.0, error=None)
        self._worker.submit(self._run, flight_id)

    def update_summary(self, flight_id: str, result: dict) -> None:
        self._write_meta(flight_id, summary=summary_of(result))

    def _run(self, flight_id: str) -> None:
        d = self.root / flight_id
        m = read_json(d / "meta.json")
        phase = m.get("phase", "prepare")
        self._write_meta(flight_id, status="running", started_utc=now(), stage="Starting")
        t0 = time.time()
        try:
            opts = FlightOptions.from_dict(m["options"])
            progress = lambda f, s: self._progress(flight_id, f, s)   # noqa: E731
            if phase == "prepare":
                info = prepare(d, opts, d / m["input_file"], progress)
                self._write_meta(flight_id, status="calibrate", progress=1.0, stage="Mark home plate and the rubber",
                                 finished_utc=now(), seconds=round(time.time() - t0, 1), **info)
            else:
                result = analyze(d, read_json(d / "meta.json"), progress)
                self._write_meta(flight_id, status="done", progress=1.0, stage="Done", finished_utc=now(),
                                 seconds=round(time.time() - t0, 1), summary=summary_of(result))
        except Exception as exc:  # noqa: BLE001
            self._fail(flight_id, exc)
        finally:
            with self._lock:
                self._live.pop(flight_id, None)
