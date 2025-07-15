# Crowd Management System v80

This version provides person and PPE detection with optional duplicate frame removal. For RTSP sources the tracker can launch FFmpeg with the `mpdecimate` filter to discard duplicate frames before they reach Python. Non‑RTSP sources use a perceptual hashing filter to skip near‑identical frames. The application is built with FastAPI and stores its configuration in Redis for easy updates. Camera tasks combine counting directions and PPE options, and line orientation is configured per camera.

## Features
- **Multiple camera sources**: Add HTTP or RTSP cameras via the settings page.
- **Object detection**: Uses YOLOv8 models for person and PPE detection. CUDA is used when available.
- **Counting and alerts**: Tracks entries/exits and can send email alerts based on customizable rules.
- **Duplicate frame filter**: Skips nearly identical frames to reduce GPU/CPU load.
- **Dashboard and reports**: Live counts, recent anomalies, and historical reports are available in the web interface.
- **Per-camera resolution**: Choose 480p, 720p, 1080p, or original when adding a camera.
- **Camera status**: Online/offline indicators appear in the Cameras page for quick troubleshooting.
- **Secure logins**: User passwords are stored as PBKDF2 hashes and verified using passlib.
- **Rotating log file**: `app.log` captures runtime logs with automatic rotation.
- **Secure logins**: User passwords are stored as PBKDF2 hashes and verified using passlib.
- **Historical reports**: A background task records per-minute counts to Redis so
  the reports page can graph occupancy over time. Log entries are stored in Redis
  sorted sets for efficient range queries.

## Installation
1. Install Python 3.10+ and Redis.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install PHP if you want to use the sample PHP pages in `public/`.

## Configuration
Edit `config.json` to set camera URLs, model paths, thresholds, and email settings. When the app starts it loads this file once, saves defaults back to disk, and stores the result in Redis. Most options can also be adjusted in the web UI under **Settings**. Key fields include:

- `stream_url` – Default video source when none are configured.
- `person_model`, `ppe_model` – Paths to YOLO models.
- `device` – `auto`, `cpu`, or `cuda:0`.
- `max_capacity` and `warn_threshold` – Occupancy limits.
- `redis_url` – Location of the Redis instance.

## Running
Launch the FastAPI application:
```bash
python3 app.py
```
Then open `http://localhost:5002` in your browser. Use the **Cameras** page to add streams (HTTP, RTSP or local webcams) and **Settings** to adjust options.

## Directory Structure
- `app.py` – FastAPI entry point.
- `core/` – Helper modules such as configuration and tracker manager.
- `modules/` – Tracking, alerts, and utilities.
- `routers/` – API routes for dashboard, settings, reports, and cameras.
- `templates/` – HTML templates rendered by FastAPI.
- `public/` – Optional PHP pages.

