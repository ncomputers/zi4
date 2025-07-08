# Crowd Management System (Version 1)

This project counts people entering and exiting through a virtual line in a video stream.
It uses [YOLOv8](https://github.com/ultralytics/ultralytics) for detection and
[DeepSORT](https://github.com/levan92/deep-sort) for tracking.

## Features
- Live video stream with bounding boxes and track IDs
- Entry/exit counts stored in Redis
- Web dashboard with real‑time updates
- Simple settings page to adjust capacity thresholds

## Running
Install dependencies and start the server:

```bash
pip install -r requirements.txt
python app.py
```

The dashboard will be available at `http://localhost:5002` by default.
