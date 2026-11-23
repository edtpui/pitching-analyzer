# syntax=docker/dockerfile:1
# Python version matches the venv the reference bundle was built in (pickled scikit-learn models).
FROM python:3.14-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PITCHLAB_HOST=0.0.0.0 PITCHLAB_PORT=8000 PITCHLAB_DATA_DIR=/data PITCHLAB_ASSETS_DIR=/srv/assets
# libGL / glib for opencv-contrib-python, EGL / GLES that mediapipe's Linux build loads even on CPU;
# ffmpeg comes from imageio-ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Runtime needs only the pose models and the built reference bundle, not the raw Driveline data.
COPY assets/models ./assets/models
COPY assets/reference/reference.json assets/reference/models.joblib ./assets/reference/
COPY pitchlab ./pitchlab
RUN useradd --create-home --uid 10001 pitchlab && mkdir /data && chown pitchlab /data

# ---- test: `docker build --target test .` runs the suite (Driveline-data tests skip: data not in image)
FROM runtime AS test
RUN pip install --no-cache-dir pytest==9.2.0 httpx==0.28.1
COPY pyproject.toml ./
COPY tests ./tests
RUN PITCHLAB_DATA_DIR=/tmp/pitchlab-test python -m pytest -q -rs tests

# ---- app: the default image
FROM runtime AS app
USER pitchlab
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["python", "-m", "pitchlab.server"]
