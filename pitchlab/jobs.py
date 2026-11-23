"""Analysis storage and a single-worker background queue.

Each analysis lives in data/analyses/<id>/ with meta.json (status, options), the uploaded file,
pose.npz, result.json, overlay.mp4 and keyframes/. One worker runs at a time: pose estimation already
uses every CPU core, so parallel jobs would only slow each other down.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import AnalysisError, Options, options_dict, read_json, run, write_json

log = logging.getLogger(__name__)
ID_RE = re.compile(r"^[0-9a-f]{12}$")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class NotFound(KeyError):
    pass


class Jobs:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self._live: dict[str, dict] = {}

    # ---- storage
    def path(self, analysis_id: str) -> Path:
        if not ID_RE.match(analysis_id or ""):
            raise NotFound(analysis_id)
        p = self.root / analysis_id
        if not (p / "meta.json").exists():
            raise NotFound(analysis_id)
        return p

    def meta(self, analysis_id: str) -> dict:
        p = self.path(analysis_id) / "meta.json"
        with self._lock:
            m = read_json(p)
            live = self._live.get(analysis_id)
        if live and m.get("status") in ("queued", "running"):
            m.update(live)
        return m

    def _write_meta(self, analysis_id: str, **changes) -> dict:
        p = self.root / analysis_id / "meta.json"
        with self._lock:
            m = read_json(p) if p.exists() else {}
            m.update(changes)
            write_json(p, m)
        return m

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and ID_RE.match(d.name) and (d / "meta.json").exists():
                try:
                    out.append(self.meta(d.name))
                except (OSError, ValueError) as exc:
                    log.warning("skipping unreadable analysis %s: %s", d.name, exc)
        return sorted(out, key=lambda m: m.get("created_utc", ""), reverse=True)

    def create(self, filename: str, suffix: str, opts: Options) -> tuple[str, Path]:
        analysis_id = uuid.uuid4().hex[:12]
        d = self.root / analysis_id
        d.mkdir(parents=True)
        self._write_meta(analysis_id, id=analysis_id, created_utc=now(), filename=filename[:200],
                         input_file=f"input{suffix}", options=options_dict(opts), status="uploading",
                         progress=0.0, stage="Uploading")
        return analysis_id, d

    def delete(self, analysis_id: str) -> None:
        m = self.meta(analysis_id)
        if m.get("status") in ("queued", "running"):
            raise RuntimeError("analysis is still running")
        shutil.rmtree(self.path(analysis_id))

    def recover(self) -> None:
        """Mark analyses that were in flight when the server stopped as failed."""
        for d in self.root.iterdir():
            mp = d / "meta.json"
            if d.is_dir() and mp.exists():
                m = read_json(mp)
                if m.get("status") in ("queued", "running", "uploading"):
                    self._write_meta(d.name, status="failed", stage="Interrupted",
                                     error="The server stopped before this analysis finished. Upload it again.")

    # ---- queue
    def submit(self, analysis_id: str) -> None:
        self._write_meta(analysis_id, status="queued", stage="Waiting in queue", progress=0.0)
        with self._lock:
            if self._executor is None:          # created lazily, and again after a shutdown
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis")
            executor = self._executor
        executor.submit(self._run, analysis_id)

    def _progress(self, analysis_id: str, frac: float, stage: str) -> None:
        with self._lock:
            self._live[analysis_id] = {"progress": round(float(frac), 4), "stage": stage}

    def _run(self, analysis_id: str) -> None:
        d = self.root / analysis_id
        m = read_json(d / "meta.json")
        self._write_meta(analysis_id, status="running", started_utc=now(), stage="Starting")
        t0 = time.time()
        try:
            opts = Options.from_dict(m["options"])
            result = run(d, opts, d / m["input_file"], lambda f, s: self._progress(analysis_id, f, s))
            pred = {p["target"]: round(p["value"], 1) for p in result.get("predictions", []) if p["reportable"]}
            self._write_meta(analysis_id, status="done", progress=1.0, stage="Done", finished_utc=now(),
                             seconds=round(time.time() - t0, 1),
                             summary={"throws": result["athlete"]["throws"],
                                      "fps": round(result["video"]["fps_effective"]),
                                      "camera_view": result["quality"]["camera_view"],
                                      "warnings": len(result["quality"]["warnings"]),
                                      "predictions": pred})
        except AnalysisError as exc:
            log.info("analysis %s failed: %s", analysis_id, exc)
            self._write_meta(analysis_id, status="failed", stage="Failed", error=str(exc), finished_utc=now())
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, full trace in the log
            log.error("analysis %s crashed:\n%s", analysis_id, traceback.format_exc())
            self._write_meta(analysis_id, status="failed", stage="Failed", finished_utc=now(),
                             error=f"Internal error ({type(exc).__name__}: {exc}). See the server log.")
        finally:
            with self._lock:
                self._live.pop(analysis_id, None)

    def shutdown(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
