# Crowd Management System Roadmap

This document outlines the high‑level plan for version 1 (`zi_v_1`).

## Goals
- Detect and track people in a video stream using YOLOv8 and DeepSORT.
- Maintain counts of entries and exits across a configurable line.
- Provide a web dashboard with live video and statistics.
- Store counts in Redis so they persist across restarts.

## Phases
1. **Prototype**
   - Implement the main tracking loop in `app.py`.
   - Provide basic HTML templates for the dashboard and settings pages.
2. **Refinement**
   - Add CSS styling and websocket updates to dashboard statistics.
   - Support configuration via a JSON file and a settings page.
3. **Deployment**
   - Document running the server with `uvicorn`.
   - Package requirements in `requirements.txt`.

Future versions will iterate on this foundation with improved UX, analytics and configuration options.
