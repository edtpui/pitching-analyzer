"""Video probing, decoding and H.264 encoding.

Decoding goes through OpenCV (per-frame timestamps, auto-rotation of phone clips). Files OpenCV cannot
decode are first transcoded with the ffmpeg binary bundled by imageio-ffmpeg. Output video is always
H.264/yuv420p MP4 with faststart, which every browser plays.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np

log = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".mts"}


class VideoError(RuntimeError):
    """The file is not a readable video."""


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str

    def as_dict(self) -> dict:
        return asdict(self)


def _fourcc(cap: cv2.VideoCapture) -> str:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    s = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))
    return s if s.isprintable() and s.strip() else "unknown"


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("OpenCV cannot open the file")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("no decodable frames")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        h, w = frame.shape[:2]
        if not (1.0 <= fps <= 2000.0):
            raise VideoError(f"implausible frame rate {fps}")
        return VideoInfo(width=w, height=h, fps=fps, frame_count=n,
                         duration_s=n / fps if n > 0 else 0.0, codec=_fourcc(cap))
    finally:
        cap.release()


def transcode_for_decoding(src: Path, dst: Path) -> Path:
    """Re-encode `src` to an H.264 MP4 OpenCV can read (keeps frame rate and every frame)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
           "-fps_mode", "passthrough", "-an", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not dst.exists():
        log.warning("ffmpeg could not decode %s: %s", src.name, proc.stderr.strip()[-800:])
        raise VideoError("the file may be incomplete (cut off while copying or uploading) or not a video at all. "
                         "Export or copy it from your phone again and re-upload.")
    return dst


def open_readable(src: Path, workdir: Path) -> tuple[Path, VideoInfo]:
    """Return a path OpenCV can decode (the original, or a transcoded copy) and its info."""
    try:
        return src, probe(src)
    except VideoError as exc:
        log.info("OpenCV failed on %s (%s); transcoding with ffmpeg", src.name, exc)
    dst = transcode_for_decoding(src, workdir / "decoded.mp4")
    return dst, probe(dst)


def scaled_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    """Size fitting in max_side, rounded to even numbers (required by yuv420p)."""
    s = min(1.0, max_side / float(max(w, h)))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def iter_frames(path: Path, max_side: int | None = None,
                start_s: float | None = None, end_s: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, BGR frame). Timestamps come from the container (VFR safe);
    if they are missing or not increasing, index / fps is used instead. Frames outside the trim
    window are skipped (frame_index still counts from the start of the file)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("OpenCV cannot open the file")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    last_t = -1.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not np.isfinite(t) or t <= last_t:
                t = i / fps if i / fps > last_t else last_t + 1.0 / fps
            last_t = t
            if end_s is not None and t > end_s:
                break
            if start_s is None or t >= start_s:
                if max_side is not None:
                    h, w = frame.shape[:2]
                    size = scaled_size(w, h, max_side)
                    if size != (w, h):
                        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                yield i, t, frame
            i += 1
    finally:
        cap.release()


class H264Writer:
    """Pipe BGR frames into ffmpeg/libx264."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        self.path = path
        self.size = size
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}",
               "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        code = self.proc.wait(timeout=600)
        if code != 0:
            raise VideoError(f"ffmpeg encoding failed: {err.strip()[-400:]}")

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            finally:
                self.proc.kill()
