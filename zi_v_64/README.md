# Crowd Management System v64

This release adds optional duplicate frame dropping to reduce load when camera views are static.

## Key Features
- HTTP or RTSP camera sources
- Person and PPE detection with YOLOv8
- Duplicate frame filter using mean absolute difference
- Configurable via `settings` page

## Usage
Install requirements and run `app.py`:
```bash
pip install -r requirements.txt
python3 app.py
```
