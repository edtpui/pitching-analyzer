import cv2
import numpy as np
import pytest

from pitchlab.swing.render import render_keyframes, render_overlay
from pitchlab.video_io import H264Writer, VideoError, iter_frames, open_readable, probe


def make_clip(path, n=48, size=(320, 240), fps=30.0):
    with H264Writer(path, size, fps) as w:
        for i in range(n):
            f = np.full((size[1], size[0], 3), 40, np.uint8)
            cv2.rectangle(f, (10 + 4 * i, 60), (50 + 4 * i, 120), (200, 120, 40), -1)
            w.write(f)
    return path


def test_writer_produces_readable_h264(tmp_path):
    p = make_clip(tmp_path / "clip.mp4")
    info = probe(p)
    assert (info.width, info.height) == (320, 240)
    assert info.fps == pytest.approx(30.0, rel=0.01)
    frames = list(iter_frames(p))
    assert len(frames) == 48
    ts = [t for _, t, _ in frames]
    assert ts[0] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(np.diff(ts), 1 / 30.0, atol=1e-3)


def test_trim_and_downscale(tmp_path):
    p = make_clip(tmp_path / "clip.mp4")
    frames = list(iter_frames(p, max_side=160, start_s=0.5, end_s=1.0))
    assert frames[0][0] == 15
    assert frames[0][2].shape[:2] == (120, 160)
    assert all(0.5 <= t <= 1.0 for _, t, _ in frames)


def test_unreadable_file_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    with pytest.raises(VideoError):
        open_readable(bad, tmp_path)


def test_overlay_and_keyframes(tmp_path):
    p = make_clip(tmp_path / "clip.mp4")
    n = 48
    pts = np.full((n, 33, 2), np.nan)
    for k in range(n):
        pts[k, 11] = (100 + k, 80)
        pts[k, 13] = (120 + k, 110)
        pts[k, 15] = (140 + k, 140)
    size = render_overlay(p, tmp_path / "overlay.mp4", np.arange(n), pts, (320, 240), "R", 30.0, 320, None, None)
    assert size == (320, 240)
    assert probe(tmp_path / "overlay.mp4").frame_count == n
    # load and foot plant on the same frame must both get a key frame
    files = render_keyframes(p, tmp_path / "kf", np.arange(n), pts, (320, 240), "L",
                             {"lift": None, "load": 20, "fp": 20, "contact": 30}, 320, None, None)
    assert set(files) == {"load", "fp", "contact"}
    assert all((tmp_path / "kf" / f).stat().st_size > 1000 for f in files.values())
