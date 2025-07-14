"""Dashboard and websocket routes."""
from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import asyncio

from core.tracker_manager import handle_status_change
from modules.utils import lock
from core.config import ANOMALY_ITEMS

router = APIRouter()

def init_context(config: dict, trackers: Dict[int, "FlowTracker"], cameras: list, redis_client):
    global cfg, trackers_map, cams, templates, redis
    cfg = config
    trackers_map = trackers
    cams = cameras
    redis = redis_client
    templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / 'templates'))

@router.get('/')
async def index(request: Request):
    in_c = sum(t.in_count for t in trackers_map.values())
    out_c = sum(t.out_count for t in trackers_map.values())
    current = in_c - out_c
    max_cap = cfg['max_capacity']
    warn_lim = max_cap * cfg['warn_threshold'] / 100
    status = 'green' if current < warn_lim else 'yellow' if current < max_cap else 'red'
    active = [c for c in cams if c.get('enabled', True)]
    anomaly_counts = {item: int(redis.get(f'{item}_count') or 0) for item in ANOMALY_ITEMS}
    return templates.TemplateResponse('dashboard.html', {
        'request': request,
        'max_capacity': max_cap,
        'status': status,
        'current': current,
        'cameras': active,
        'cfg': cfg,
        'anomaly_counts': anomaly_counts,
    })

@router.get('/video_feed/{cam_id}')
async def video_feed(cam_id: int):
    tr = trackers_map.get(cam_id)
    if not tr:
        return HTMLResponse('Not found', status_code=404)

    async def gen():
        while True:
            with lock:
                frame = tr.output_frame
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            _, buf = cv2.imencode('.jpg', frame)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            await asyncio.sleep(1 / tr.fps)
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@router.websocket('/ws/stats')
async def ws_stats(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            in_c = sum(t.in_count for t in trackers_map.values())
            out_c = sum(t.out_count for t in trackers_map.values())
            current = in_c - out_c
            max_cap = cfg['max_capacity']
            warn_lim = max_cap * cfg['warn_threshold'] / 100
            status = 'green' if current < warn_lim else 'yellow' if current < max_cap else 'red'
            handle_status_change(status, redis)
            anomaly_counts = {item: int(redis.get(f'{item}_count') or 0) for item in ANOMALY_ITEMS}
            await ws.send_json({'in_count': in_c, 'out_count': out_c, 'current': current, 'max_capacity': max_cap, 'status': status, 'anomaly_counts': anomaly_counts})
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

@router.get('/latest_images')
async def latest_images(status: str = 'no_helmet', count: int = 5):
    entries = redis.lrange('ppe_logs', -1000, -1)
    imgs = []
    for item in reversed(entries):
        e = json.loads(item)
        if e.get('status') == status and e.get('path'):
            fname = os.path.basename(e['path'])
            imgs.append(f'/snapshots/{fname}')
            if len(imgs) >= count:
                break
    return {'images': imgs}
