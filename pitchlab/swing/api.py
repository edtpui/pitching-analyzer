"""HTTP routes for swing analyses, mounted by the app under /api/swings."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import MAX_UPLOAD_MB
from ..jobs import NotFound
from ..video_io import ALLOWED_EXTENSIONS
from .coaching import LLMUnavailable, claude_report
from .jobs import SwingJobs
from .pipeline import Options, read_json, recompute_with_events, write_json
from .reference import ReferenceMissing, get_reference

KEYFRAME_RE = re.compile(r"^(lift|load|fp|contact)\.jpg$")
CHUNK = 1 << 20


class EventEdit(BaseModel):
    lift: int | None = Field(default=None, ge=0)
    load: int | None = Field(default=None, ge=0)
    fp: int | None = Field(default=None, ge=0)
    contact: int | None = Field(default=None, ge=0)


def _num(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise HTTPException(422, f"not a number: {v!r}") from None


def reference_status() -> dict:
    """For /api/health: whether the hitting reference bundle is built."""
    try:
        ref = get_reference()
        return {"ok": True, "built_utc": ref.bundle["built_utc"], "models": ref.models is not None}
    except ReferenceMissing as exc:
        return {"ok": False, "error": str(exc)}


def make_router(jobs: SwingJobs) -> APIRouter:
    r = APIRouter(prefix="/api/swings", tags=["swings"])

    def meta_or_404(swing_id: str) -> dict:
        try:
            return jobs.meta(swing_id)
        except NotFound:
            raise HTTPException(404, "swing analysis not found") from None

    @r.get("/reference")
    def reference() -> dict:
        try:
            return get_reference().summary()
        except ReferenceMissing as exc:
            raise HTTPException(503, str(exc)) from None

    @r.get("")
    def list_swings() -> list[dict]:
        return jobs.list()

    @r.post("", status_code=202)
    async def create_swing(
        file: UploadFile = File(...),
        bats: str = Form("auto"), alignment: str = Form("square"), model: str = Form("full"),
        capture_fps: str | None = Form(None), trim_start_s: str | None = Form(None), trim_end_s: str | None = Form(None),
        height_cm: str | None = Form(None), weight_kg: str | None = Form(None), bat_speed_mph: str | None = Form(None),
        level: str | None = Form(None), label: str = Form(""),
    ) -> dict:
        try:
            get_reference()
        except ReferenceMissing as exc:
            raise HTTPException(503, str(exc)) from None
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"unsupported file type {suffix or '(none)'}; use {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        try:
            opts = Options(bats=bats, alignment=alignment, model=model, capture_fps=_num(capture_fps),
                           trim_start_s=_num(trim_start_s), trim_end_s=_num(trim_end_s), height_cm=_num(height_cm),
                           weight_kg=_num(weight_kg), bat_speed_mph=_num(bat_speed_mph),
                           level=level or None, label=label).validate()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

        swing_id, d = jobs.create(name, suffix, opts)
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
        jobs.submit(swing_id)
        return {"id": swing_id}

    @r.get("/{swing_id}")
    def get_swing(swing_id: str) -> dict:
        meta = meta_or_404(swing_id)
        out = {"meta": meta}
        rp = jobs.path(swing_id) / "result.json"
        if meta.get("status") == "done" and rp.exists():
            out["result"] = read_json(rp)
        return out

    @r.delete("/{swing_id}", status_code=204)
    def delete_swing(swing_id: str) -> None:
        meta_or_404(swing_id)
        try:
            jobs.delete(swing_id)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None

    @r.post("/{swing_id}/events")
    def edit_events(swing_id: str, body: EventEdit) -> dict:
        meta = meta_or_404(swing_id)
        if meta.get("status") != "done":
            raise HTTPException(409, "analysis is not finished")
        d = jobs.path(swing_id)
        try:
            result = recompute_with_events(d, Options.from_dict(meta["options"]), d / meta["input_file"],
                                           body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        jobs.update_summary(swing_id, result)
        return {"meta": jobs.meta(swing_id), "result": result}

    @r.post("/{swing_id}/report")
    def make_report(swing_id: str) -> dict:
        meta = meta_or_404(swing_id)
        if meta.get("status") != "done":
            raise HTTPException(409, "analysis is not finished")
        rp = jobs.path(swing_id) / "result.json"
        result = read_json(rp)
        try:
            report = claude_report(result)
        except LLMUnavailable as exc:
            raise HTTPException(503, f"Claude report unavailable: {exc}") from None
        result["llm_report"] = report
        write_json(rp, result)
        return report

    @r.get("/{swing_id}/overlay.mp4")
    def overlay(swing_id: str) -> FileResponse:
        meta_or_404(swing_id)
        p = jobs.path(swing_id) / "overlay.mp4"
        if not p.exists():
            raise HTTPException(404, "overlay not rendered")
        return FileResponse(p, media_type="video/mp4")

    @r.get("/{swing_id}/keyframes/{name}")
    def keyframe(swing_id: str, name: str) -> FileResponse:
        meta_or_404(swing_id)
        if not KEYFRAME_RE.match(name):
            raise HTTPException(404, "unknown key frame")
        p = jobs.path(swing_id) / "keyframes" / name
        if not p.exists():
            raise HTTPException(404, "key frame not rendered")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    return r
