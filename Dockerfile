# Deploys the Flask app (server.py) as a container -- built for
# Hugging Face Spaces' "Docker" SDK, which expects the app to listen on
# port 7860, but this works the same on any Docker host (a VPS,
# Cloud Run, etc.) if that's needed later.

FROM python:3.11-slim

WORKDIR /app

# System libraries OpenCV needs at runtime even with the headless wheel
# (video/image codec support) -- opencv-python-headless doesn't need
# the GUI libs (Qt etc), just these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

COPY . .

# Directories the app writes to (config.py also creates these at
# import time, but pre-creating them here avoids any doubt about
# write permissions in a fresh container).
RUN mkdir -p uploads output/plots output/csv

ENV PORT=7860
EXPOSE 7860

CMD gunicorn server:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 600
