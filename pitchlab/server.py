"""FastAPI app: upload clips, follow progress, read results, correct events, request a Claude report.

Run:  .venv\\Scripts\\python.exe -m pitchlab.server      (binds 127.0.0.1:8091 by default)
"""
from __future__ import annotations

import base64
import binascii
import hmac
import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .coaching import LLMUnavailable, claude_report, has_credentials
from .config import ANALYSES_DIR, ANTHROPIC_MODEL, HOST, MAX_UPLOAD_MB, MODELS_DIR, PASSWORD, PORT
from .jobs import Jobs, NotFound
from .pipeline import Options, read_json, recompute_with_events, write_json
from .reference import ReferenceMissing, get_reference
from .video_io import ALLOWED_EXTENSIONS

log = logging.getLogger("pitchlab")
STATIC_DIR = Path(__file__).parent / "static"
KEYFRAME_RE = re.compile(r"^(pkh|fp|mer|br)\.jpg$")
CHUNK = 1 << 20

jobs = Jobs(ANALYSES_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    jobs.recover()
    try:
        get_reference()
    except ReferenceMissing as exc:
        log.error("%s", exc)
    yield
    jobs.shutdown()


app = FastAPI(title="Pitching mechanics analyzer", version=__version__, lifespan=lifespan)


def _authorized(header: str) -> bool:
    if not header.startswith("Basic "):
        return False
    try:
        _, _, password = base64.b64decode(header[6:], validate=True).decode("utf-8").partition(":")
    except (binascii.Error, UnicodeDecodeError):
        return False
    return hmac.compare_digest(password.encode(), PASSWORD.encode())


@app.middleware("http")
async def require_password(request: Request, call_next):
    """When PITCHLAB_PASSWORD is set (always, for the public container), every route except the health
    probe needs HTTP Basic auth with that password; the user name is ignored. Browsers re-send the
    credentials on fetches, uploads and video range requests by themselves."""
    if PASSWORD and request.url.path != "/api/health" and not _authorized(request.headers.get("authorization", "")):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Pitch Mechanics Analyzer", charset="UTF-8"'})
    return await call_next(request)


def _meta_or_404(analysis_id: str) -> dict:
    try:
        return jobs.meta(analysis_id)
    except NotFound:
        raise HTTPException(404, "analysis not found") from None


def _num(v: str | None) -> float | None:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise HTTPException(422, f"not a number: {v!r}") from None


@app.get("/api/health")
def health() -> dict:
    out = {"version": __version__, "pose_models": sorted(p.stem.replace("pose_landmarker_", "")
                                                         for p in MODELS_DIR.glob("pose_landmarker_*.task"))}
    try:
        ref = get_reference()
        out["reference"] = {"ok": True, "built_utc": ref.bundle["built_utc"], "models": ref.models is not None}
    except ReferenceMissing as exc:
        out["reference"] = {"ok": False, "error": str(exc)}
    out["llm"] = _llm_status()
    return out


def _llm_status() -> dict:
    try:
        import anthropic
        configured = has_credentials(anthropic.Anthropic())
        return {"configured": configured, "model": ANTHROPIC_MODEL,
                "reason": None if configured else "no ANTHROPIC_API_KEY, auth token or profile found"}
    except Exception as exc:  # noqa: BLE001 - any failure means "not configured"
        return {"configured": False, "model": ANTHROPIC_MODEL, "reason": str(exc)[:200]}


@app.get("/api/reference")
def reference() -> dict:
    try:
        return get_reference().summary()
    except ReferenceMissing as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/api/analyses")
def list_analyses() -> list[dict]:
    return jobs.list()


@app.post("/api/analyses", status_code=202)
async def create_analysis(
    file: UploadFile = File(...),
    throws: str = Form("auto"), alignment: str = Form("square"), model: str = Form("full"),
    capture_fps: str | None = Form(None), trim_start_s: str | None = Form(None), trim_end_s: str | None = Form(None),
    height_cm: str | None = Form(None), weight_kg: str | None = Form(None), velocity_mph: str | None = Form(None),
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
        opts = Options(throws=throws, alignment=alignment, model=model, capture_fps=_num(capture_fps),
                       trim_start_s=_num(trim_start_s), trim_end_s=_num(trim_end_s), height_cm=_num(height_cm),
                       weight_kg=_num(weight_kg), velocity_mph=_num(velocity_mph),
                       level=level or None, label=label).validate()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None

    analysis_id, d = jobs.create(name, suffix, opts)
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
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        raise
    finally:
        await file.close()
    meta = read_json(d / "meta.json")
    meta["size_bytes"] = written
    write_json(d / "meta.json", meta)
    jobs.submit(analysis_id)
    return {"id": analysis_id}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict:
    meta = _meta_or_404(analysis_id)
    out = {"meta": meta}
    rp = jobs.path(analysis_id) / "result.json"
    if meta.get("status") == "done" and rp.exists():
        out["result"] = read_json(rp)
    return out


@app.delete("/api/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str) -> None:
    _meta_or_404(analysis_id)
    try:
        jobs.delete(analysis_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None


class EventEdit(BaseModel):
    pkh: int | None = Field(default=None, ge=0)
    fp: int | None = Field(default=None, ge=0)
    mer: int | None = Field(default=None, ge=0)
    br: int | None = Field(default=None, ge=0)


@app.post("/api/analyses/{analysis_id}/events")
def edit_events(analysis_id: str, body: EventEdit) -> dict:
    meta = _meta_or_404(analysis_id)
    if meta.get("status") != "done":
        raise HTTPException(409, "analysis is not finished")
    d = jobs.path(analysis_id)
    try:
        result = recompute_with_events(d, Options.from_dict(meta["options"]), d / meta["input_file"],
                                       body.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"meta": jobs.meta(analysis_id), "result": result}


@app.post("/api/analyses/{analysis_id}/report")
def make_report(analysis_id: str) -> dict:
    meta = _meta_or_404(analysis_id)
    if meta.get("status") != "done":
        raise HTTPException(409, "analysis is not finished")
    rp = jobs.path(analysis_id) / "result.json"
    result = read_json(rp)
    try:
        report = claude_report(result)
    except LLMUnavailable as exc:
        raise HTTPException(503, f"Claude report unavailable: {exc}") from None
    result["llm_report"] = report
    write_json(rp, result)
    return report


@app.get("/api/analyses/{analysis_id}/overlay.mp4")
def overlay(analysis_id: str) -> FileResponse:
    _meta_or_404(analysis_id)
    p = jobs.path(analysis_id) / "overlay.mp4"
    if not p.exists():
        raise HTTPException(404, "overlay not rendered")
    return FileResponse(p, media_type="video/mp4")


@app.get("/api/analyses/{analysis_id}/keyframes/{name}")
def keyframe(analysis_id: str, name: str) -> FileResponse:
    _meta_or_404(analysis_id)
    if not KEYFRAME_RE.match(name):
        raise HTTPException(404, "unknown key frame")
    p = jobs.path(analysis_id) / "keyframes" / name
    if not p.exists():
        raise HTTPException(404, "key frame not rendered")
    return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.exception_handler(NotFound)
async def not_found(_: Request, __: NotFound) -> JSONResponse:
    return JSONResponse({"detail": "analysis not found"}, status_code=404)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class RevalidatedStatic(StaticFiles):
    """Static files with Cache-Control: no-cache, so a browser revalidates (cheap 304 via ETag) instead of
    running a stale app.js/app.css after an update."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", RevalidatedStatic(directory=STATIC_DIR), name="static")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
