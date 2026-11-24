"""HTTP routes for ball-flight analyses (mounted by pitchlab.server)."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from ..config import MAX_UPLOAD_MB
from ..pipeline import read_json, write_json
from ..video_io import ALLOWED_EXTENSIONS
from .geometry import CalibrationError
from .jobs import FlightJobs, summary_of
from .pipeline import FlightError, FlightOptions, calibrate_flight, frame_image, recompute_release

CHUNK = 1 << 20


def _num(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise HTTPException(422, f"not a number: {v!r}") from None


class CalibrationBody(BaseModel):
    plate_px: list[list[float]] | None = None
    rubber_px: list[list[float]] | None = None
    image: str | int = "background"
    camera_height_ft: float | None = Field(default=None, ge=0.5, le=30)
    camera_ahead_ft: float | None = Field(default=None, ge=-30, le=70)    # tape: ahead of the rubber, toward home
    camera_side_ft: float | None = Field(default=None, ge=-30, le=30)     # tape: + first-base side, - third-base side
    distance_ft: float | None = Field(default=None, ge=30, le=70)
    mound_in: float | None = Field(default=None, ge=0, le=18)
    copy_from: str | None = None


class ReleaseBody(BaseModel):
    frame: int | None = Field(default=None, ge=0)


def make_router(jobs: FlightJobs) -> APIRouter:
    r = APIRouter(prefix="/api/flights")

    def meta_or_404(fid: str) -> dict:
        try:
            return jobs.meta(fid)
        except KeyError:
            raise HTTPException(404, "flight analysis not found") from None

    def busy(meta: dict) -> bool:
        return meta.get("status") in ("uploading", "queued", "running")

    @r.get("")
    def list_flights() -> list[dict]:
        return jobs.list()

    @r.post("", status_code=202)
    async def create_flight(
        file: UploadFile = File(...), throws: str = Form("R"), distance_ft: str | None = Form(None),
        mound_in: str | None = Form(None), zone_bottom_in: str | None = Form(None), zone_top_in: str | None = Form(None),
        capture_fps: str | None = Form(None), trim_start_s: str | None = Form(None), trim_end_s: str | None = Form(None),
        pitcher_height_in: str | None = Form(None), altitude_ft: str | None = Form(None), label: str = Form(""),
    ) -> dict:
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"unsupported file type {suffix or '(none)'}; use {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        try:
            raw = dict(distance_ft=distance_ft, mound_in=mound_in, zone_bottom_in=zone_bottom_in, zone_top_in=zone_top_in,
                       capture_fps=capture_fps, trim_start_s=trim_start_s, trim_end_s=trim_end_s,
                       pitcher_height_in=pitcher_height_in, altitude_ft=altitude_ft)
            kw = {k: v for k, v in ((k, _num(v)) for k, v in raw.items()) if v is not None}
            opts = FlightOptions(throws=throws, label=label, **kw).validate()
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        fid, d = jobs.create(name, suffix, opts)
        dest = d / f"input{suffix}"
        limit = MAX_UPLOAD_MB * CHUNK
        written = 0
        try:
            with dest.open("wb") as out:
                while chunk := await file.read(CHUNK):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
            if written == 0:
                raise HTTPException(422, "the uploaded file is empty")
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        finally:
            await file.close()
        meta = read_json(d / "meta.json")
        meta["size_bytes"] = written
        write_json(d / "meta.json", meta)
        jobs.submit(fid, "prepare")
        return {"id": fid}

    @r.get("/{fid}")
    def get_flight(fid: str) -> dict:
        meta = meta_or_404(fid)
        d = jobs.path(fid)
        out = {"meta": meta}
        if (d / "calibration.json").exists():
            out["calibration"] = read_json(d / "calibration.json")
        if meta.get("status") == "done" and (d / "result.json").exists():
            out["result"] = read_json(d / "result.json")
        return out

    @r.delete("/{fid}", status_code=204)
    def delete_flight(fid: str) -> None:
        meta_or_404(fid)
        try:
            jobs.delete(fid)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @r.get("/{fid}/background.jpg")
    def background(fid: str) -> FileResponse:
        meta_or_404(fid)
        p = jobs.path(fid) / "background.jpg"
        if not p.exists():
            raise HTTPException(404, "background not ready")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @r.get("/{fid}/frames/{index}.jpg")
    def frame(fid: str, index: int) -> Response:
        meta = meta_or_404(fid)
        if busy(meta) and meta.get("phase") == "prepare":
            raise HTTPException(409, "the clip is still being prepared")
        if index < 0:
            raise HTTPException(404, "no such frame")
        try:
            img = frame_image(jobs.path(fid), meta, index)
        except FlightError as exc:
            raise HTTPException(404, str(exc)) from None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @r.post("/{fid}/calibration")
    def calibration(fid: str, body: CalibrationBody) -> dict:
        meta = meta_or_404(fid)
        if busy(meta):
            raise HTTPException(409, "wait until the clip is ready")
        d = jobs.path(fid)
        payload = body.model_dump()
        if body.copy_from:
            try:
                src = jobs.path(body.copy_from) / "calibration.json"
            except KeyError:
                raise HTTPException(404, "the analysis to copy from was not found") from None
            if not src.exists():
                raise HTTPException(422, "that analysis has no calibration")
            payload = read_json(src) | {k: v for k, v in payload.items()
                                        if k in ("camera_height_ft", "camera_ahead_ft", "camera_side_ft") and v is not None}
            payload["image"] = "background"
        elif not body.plate_px:
            raise HTTPException(422, "mark all five corners of home plate")
        try:
            return calibrate_flight(d, meta, payload)
        except (CalibrationError, FlightError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from None

    @r.post("/{fid}/analyze", status_code=202)
    def analyze(fid: str) -> dict:
        meta = meta_or_404(fid)
        if busy(meta):
            raise HTTPException(409, "already running")
        d = jobs.path(fid)
        if not (d / "background.png").exists():
            raise HTTPException(409, "the clip was not prepared; upload it again")
        if not (d / "calibration.json").exists():
            raise HTTPException(409, "mark home plate and the rubber first")
        jobs.submit(fid, "analyze")
        return {"id": fid}

    @r.post("/{fid}/release")
    def release(fid: str, body: ReleaseBody) -> dict:
        meta = meta_or_404(fid)
        if meta.get("status") != "done":
            raise HTTPException(409, "analysis is not finished")
        try:
            result = recompute_release(jobs.path(fid), meta, body.frame)
        except (ValueError, FlightError) as exc:
            raise HTTPException(422, str(exc)) from None
        jobs.update_summary(fid, result)
        return {"meta": jobs.meta(fid), "result": result}

    @r.get("/{fid}/tracer.mp4")
    def tracer(fid: str) -> FileResponse:
        meta_or_404(fid)
        p = jobs.path(fid) / "tracer.mp4"
        if not p.exists():
            raise HTTPException(404, "tracer video not rendered")
        return FileResponse(p, media_type="video/mp4")

    @r.get("/{fid}/seams.jpg")
    def seams(fid: str) -> FileResponse:
        meta_or_404(fid)
        p = jobs.path(fid) / "seams.jpg"
        if not p.exists():
            raise HTTPException(404, "no seam fit")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    return r


__all__ = ["make_router", "summary_of"]
