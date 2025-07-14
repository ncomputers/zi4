# Crowd Management System (Version 2)

This version improves the dashboard with Bootstrap styling and live charts using Chart.js.
It exposes the same FastAPI endpoints but offers a more user‑friendly interface for monitoring
people entering and exiting a space.

## Features
- Bootstrap based dashboard with responsive layout
- Live video stream and real‑time counts via WebSocket
- Settings page to configure capacity limits and warning thresholds
- Line chart visualizing entry/exit counts over time

## Running
```bash
pip install -r requirements.txt
python app.py
```

The application reads options from `config.json`.
