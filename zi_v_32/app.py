#!/usr/bin/env python3
"""Crowd management system version 32."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import redis
import smtplib
import uvicorn
from email.message import EmailMessage
import io
import csv

# Globals
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
lock = threading.Lock()

# Object classes available for counting
PPE_ITEMS = ["helmet", "glove", "shoes", "goggles", "mask"]
ANOMALY_ITEMS = ["no_helmet", "no_glove", "no_shoes", "no_goggles", "no_mask"]
COUNT_GROUPS = {
    "person": ["person"],
    "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle"],
    "animal": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
}
AVAILABLE_CLASSES = (
    PPE_ITEMS
    + ANOMALY_ITEMS
    + [c for cl in COUNT_GROUPS.values() for c in cl]
)
config_path: str
redis_client: redis.Redis
trackers: dict[int, FlowTracker] = {}
cameras: list
last_status: str | None = None


def sync_detection_classes(cfg: dict) -> None:
    """Build class lists for object and PPE detection."""
    object_classes: list[str] = []
    count_classes: list[str] = []
    for group in cfg.get("track_objects", ["person"]):
        count_classes.extend(COUNT_GROUPS.get(group, [group]))
    object_classes.extend(count_classes)
    ppe_classes: list[str] = []
    for item in cfg.get("track_ppe", []):
        ppe_classes.append(item)
        neg = f"no_{item}"
        if neg in AVAILABLE_CLASSES:
            ppe_classes.append(neg)
    cfg["object_classes"] = object_classes
    cfg["ppe_classes"] = ppe_classes
    cfg["count_classes"] = count_classes


def load_config(path: str, r: redis.Redis) -> dict:
    if os.path.exists(path):
        data = json.load(open(path))
        data.setdefault("track_ppe", [])
        data.setdefault("alert_anomalies", [])
        data.setdefault("track_objects", ["person"])
        data.setdefault("line_orientation", "vertical")
        data.setdefault("person_model", "yolov8n.pt")
        data.setdefault("ppe_model", "best_hlmt.pt")
        sync_detection_classes(data)
        r.set("config", json.dumps(data))
        return data
    raise FileNotFoundError(path)


def save_config(cfg: dict, path: str, r: redis.Redis) -> None:
    sync_detection_classes(cfg)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    r.set("config", json.dumps(cfg))


def load_cameras(r: redis.Redis, default_url: str) -> list:
    """Load camera list from Redis or create a default one."""
    data = r.get("cameras")
    if data:
        try:
            cams = json.loads(data)
            # drop legacy class information
            for cam in cams:
                cam.pop("classes", None)
            return cams
        except json.JSONDecodeError:
            pass
    cams = [{
        "id": 1,
        "name": "Camera1",
        "url": default_url,
        "mode": "both",
        "enabled": True,
    }]
    r.set("cameras", json.dumps(cams))
    return cams


def save_cameras(cams: list, r: redis.Redis) -> None:
    r.set("cameras", json.dumps(cams))


def reset_counts():
    """Reset all camera counts."""
    for tr in trackers.values():
        tr.in_count = 0
        tr.out_count = 0
        tr.tracks.clear()
        tr.prev_date = date.today()
        tr.redis.mset({tr.key_in: 0, tr.key_out: 0, tr.key_date: tr.prev_date.isoformat()})
    logger.info("Counts reset")


def log_counts():
    ts = int(time.time())
    in_c = sum(t.in_count for t in trackers.values())
    out_c = sum(t.out_count for t in trackers.values())
    entry = json.dumps({"ts": ts, "in": in_c, "out": out_c})
    redis_client.rpush("history", entry)

async def count_log_loop():
    while True:
        log_counts()
        await asyncio.sleep(60)


def send_email(subject: str, message: str, to_list: list[str], image: bytes | None = None):
    cfg = config.get("email", {})
    host = cfg.get("smtp_host")
    if not host:
        return
    to_addrs = [a.strip() for a in to_list if a.strip()]
    if cfg.get("cc"):
        to_addrs += [a.strip() for a in cfg["cc"].split(',') if a.strip()]
    if cfg.get("bcc"):
        to_addrs += [a.strip() for a in cfg["bcc"].split(',') if a.strip()]
    msg = EmailMessage()
    msg["From"] = cfg.get("from_addr")
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.set_content(message)
    if image:
        msg.add_attachment(image, maintype="image", subtype="jpeg", filename="alert.jpg")
    try:
        with smtplib.SMTP(host, cfg.get("smtp_port", 587)) as s:
            if cfg.get("use_tls", True):
                s.starttls()
            if cfg.get("smtp_user"):
                s.login(cfg.get("smtp_user"), cfg.get("smtp_pass", ""))
            s.send_message(msg)
    except Exception as exc:
        logger.error("Email send failed: %s", exc)


def handle_status_change(status: str):
    global last_status
    if status == last_status:
        return
    last_status = status
    cfg = config.get("email", {})
    to_field = "to_yellow" if status == "yellow" else "to_red" if status == "red" else None
    if not to_field:
        return
    to_addrs = [a.strip() for a in cfg.get(to_field, "").split(',') if a.strip()]
    if not to_addrs:
        return
    threading.Thread(
        target=send_email,
        args=(f"Crowd Alert - {status.title()}", f"Current status: {status}", to_addrs),
        daemon=True,
    ).start()


async def auto_reset_loop():
    while True:
        sched = config.get("reset_schedule", "never")
        last = float(redis_client.get("last_reset") or 0)
        now = time.time()
        delta = now - last
        need = False
        if sched == "hourly" and delta >= 3600:
            need = True
        elif sched == "daily" and delta >= 86400:
            need = True
        elif sched == "weekly" and delta >= 604800:
            need = True
        elif sched == "monthly" and delta >= 2592000:
            need = True
        if need:
            reset_counts()
            redis_client.set("last_reset", now)
        await asyncio.sleep(60)


def start_tracker(cam: dict) -> FlowTracker:
    """Create and start a FlowTracker for a camera."""
    tr = FlowTracker(
        cam["id"],
        cam["url"],
        config.get("object_classes", ["person"]),
        config,
    )
    trackers[cam["id"]] = tr
    threading.Thread(target=tr.capture_loop, daemon=True).start()
    threading.Thread(target=tr.process_loop, daemon=True).start()
    return tr


def stop_tracker(cam_id: int) -> None:
    tr = trackers.get(cam_id)
    if tr:
        tr.running = False
        trackers.pop(cam_id, None)


class FlowTracker:
    """Tracks entry and exit counts using YOLOv8 and DeepSORT."""

    def __init__(self, cam_id: int, src: str, classes: list[str], cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.cam_id = cam_id
        self.src = src
        self.classes = classes
        self.count_classes = cfg.get("count_classes", [])
        self.alert_anomalies = cfg.get("alert_anomalies", [])
        self.line_orientation = cfg.get("line_orientation", "vertical")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading person model {self.person_model} on {self.device}")
        self.model_person = YOLO(self.person_model)
        logger.info(f"Loading PPE model {self.ppe_model} on {self.device}")
        self.model_ppe = YOLO(self.ppe_model)
        if self.device.startswith("cuda"):
            self.model_person.model.to(self.device)
            self.model_ppe.model.to(self.device)
            torch.backends.cudnn.benchmark = True
        self.tracker = DeepSort(max_age=5)
        self.frame_queue = queue.Queue(maxsize=10)
        self.tracks = {}
        self.redis = redis.Redis.from_url(self.redis_url)
        key_prefix = f"cam:{self.cam_id}:"
        self.key_in = key_prefix + "in"
        self.key_out = key_prefix + "out"
        self.key_date = key_prefix + "date"
        self.in_count = int(self.redis.get(self.key_in) or 0)
        self.out_count = int(self.redis.get(self.key_out) or 0)
        stored_date = self.redis.get(self.key_date)
        self.prev_date = (
            date.fromisoformat(stored_date.decode()) if stored_date else date.today()
        )
        self.redis.mset({
            self.key_in: self.in_count,
            self.key_out: self.out_count,
            self.key_date: self.prev_date.isoformat(),
        })
        self.output_frame = None
        self.running = True

    def update_cfg(self, cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)
        # update object classes if provided
        if "object_classes" in cfg:
            self.classes = cfg["object_classes"]
        if "count_classes" in cfg:
            self.count_classes = cfg["count_classes"]
        if "alert_anomalies" in cfg:
            self.alert_anomalies = cfg["alert_anomalies"]
        if "line_orientation" in cfg:
            self.line_orientation = cfg["line_orientation"]
        if "person_model" in cfg and cfg["person_model"] != getattr(self, "person_model", None):
            self.person_model = cfg["person_model"]
            self.model_person = YOLO(self.person_model)
            if self.device.startswith("cuda"):
                self.model_person.model.to(self.device)
        if "ppe_model" in cfg and cfg["ppe_model"] != getattr(self, "ppe_model", None):
            self.ppe_model = cfg["ppe_model"]
            self.model_ppe = YOLO(self.ppe_model)
            if self.device.startswith("cuda"):
                self.model_ppe.model.to(self.device)

    def capture_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                logger.error(f"Cannot open stream: {self.src}")
                time.sleep(self.retry_interval)
                continue
            logger.info(f"Stream opened: {self.src}")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Lost stream, retry in {self.retry_interval}s")
                    break
                if self.frame_queue.full():
                    _ = self.frame_queue.get()
                self.frame_queue.put(frame)
            cap.release()
            time.sleep(self.retry_interval)

    def process_loop(self):
        idx = 0
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            idx += 1
            if date.today() != self.prev_date:
                self.in_count = 0
                self.out_count = 0
                self.tracks.clear()
                self.prev_date = date.today()
                self.redis.mset({
                    self.key_in: self.in_count,
                    self.key_out: self.out_count,
                    self.key_date: self.prev_date.isoformat(),
                })
                logger.info("Daily counts reset")
            if self.skip_frames and idx % self.skip_frames:
                continue
            res = self.model_person.predict(frame, device=self.device, verbose=False)[0]
            h, w = frame.shape[:2]
            if self.line_orientation == 'horizontal':
                line_pos = int(h * self.line_ratio)
                cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)
            else:
                line_pos = int(w * self.line_ratio)
                cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), 2)
            dets = []
            for *xyxy, conf, cls in res.boxes.data.tolist():
                label = self.model_person.names[int(cls)] if isinstance(self.model_person.names, dict) else self.model_person.names[int(cls)]
                if label in self.classes and conf >= self.conf_thresh:
                    dets.append([
                        [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                        conf,
                        label,
                    ])
            tracks = self.tracker.update_tracks(dets, frame=frame)
            now = time.time()
            active_ids = set()
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                active_ids.add(tid)
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if self.line_orientation == 'horizontal':
                    zone = 'top' if cy < line_pos else 'bottom'
                else:
                    zone = 'left' if cx < line_pos else 'right'
                label = getattr(tr, 'det_class', None)
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        'zone': zone,
                        'cx': cx,
                        'time': now,
                        'last': None,
                        'alerted': False,
                        'label': label,
                        'best_conf': 0.0,
                        'best_img': None,
                    }
                prev = self.tracks[tid]
#<<<<<<< ftwijc-codex/create-full-crowd-management-system
                conf = getattr(tr, 'det_conf', 0) or 0

                if label == 'person' and conf > prev.get('best_conf', 0):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        prev['best_conf'] = conf
                        prev['best_img'] = crop.copy()
                if label is not None:
                    prev['label'] = label
                if zone != prev['zone'] and abs(cx - prev['cx']) > self.v_thresh and now - prev['time'] > self.debounce:
                    direction = None
                    if self.line_orientation == 'horizontal':
                        if prev['zone'] == 'top' and zone == 'bottom':
                            direction = 'Entering'
                        elif prev['zone'] == 'bottom' and zone == 'top':
                            direction = 'Exiting'
                    else:
                        if prev['zone'] == 'left' and zone == 'right':
                            direction = 'Entering'
                        elif prev['zone'] == 'right' and zone == 'left':
                            direction = 'Exiting'
                    if direction and prev.get('label') in self.count_classes:
                        if prev['last'] is None:
                            if direction == 'Entering':
                                self.in_count += 1
                            else:
                                self.out_count += 1
                            self.redis.mset({self.key_in: self.in_count, self.key_out: self.out_count})
                            prev['last'] = direction
                            logger.info(f"{direction} ID{tid}: In={self.in_count} Out={self.out_count}")
                        elif prev['last'] != direction:
                            if prev['last'] == 'Entering':
                                self.in_count -= 1
                            else:
                                self.out_count -= 1
                            self.redis.mset({self.key_in: self.in_count, self.key_out: self.out_count})
                            prev['last'] = None
                            logger.info(f"Reversed flow for ID{tid}")
                        prev['time'] = now
                prev['zone'], prev['cx'] = zone, cx
                prev['last_seen'] = now
                color = (0, 255, 0) if zone == 'right' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID{tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # process tracks that have disappeared
            gone_ids = [tid for tid in list(self.tracks.keys()) if tid not in active_ids]
            for tid in gone_ids:
                info = self.tracks.pop(tid)
                img = info.get('best_img')
                anomaly = None
                if img is not None and img.size:
                    pres = self.model_ppe.predict(img, device=self.device, verbose=False)[0]
                    for *pxyxy, pconf, pcls in pres.boxes.data.tolist():
                        plabel = self.model_ppe.names[int(pcls)] if isinstance(self.model_ppe.names, dict) else self.model_ppe.names[int(pcls)]
                        if pconf >= self.conf_thresh and plabel in self.ppe_classes:
                            anomaly = plabel
                            break
                if anomaly and anomaly in self.alert_anomalies and not info.get('alerted'):
                    snap = img if img is not None else frame
                    _, buf = cv2.imencode('.jpg', snap)
                    to_emails = config.get('email', {}).get('ppe_to', '')
                    recipients = [a.strip() for a in to_emails.split(',') if a.strip()]
                    threading.Thread(
                        target=send_email,
                        args=(f'PPE Alert: {anomaly}', f'Camera {self.cam_id} detected {anomaly}', recipients, buf.tobytes()),
                        daemon=True,
                    ).start()

            cv2.putText(frame, f"Entering: {self.in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exiting: {self.out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with lock:
                self.output_frame = frame.copy()
            time.sleep(1 / self.fps)


app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


@app.get("/")
async def index(request: Request):
    in_c = sum(t.in_count for t in trackers.values())
    out_c = sum(t.out_count for t in trackers.values())
    current = in_c - out_c
    max_cap = config["max_capacity"]
    warn_lim = max_cap * config["warn_threshold"] / 100
    status = "green" if current < warn_lim else "yellow" if current < max_cap else "red"
    active = [c for c in cameras if c.get("enabled", True)]
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "max_capacity": max_cap,
            "status": status,
            "current": current,
            "cameras": active,
        },
    )


@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    tr = trackers.get(cam_id)
    if not tr:
        return HTMLResponse("Not found", status_code=404)

    async def gen():
        while True:
            with lock:
                frame = tr.output_frame
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            _, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            await asyncio.sleep(1/tr.fps)
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket('/ws/stats')
async def ws_stats(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            in_c = sum(t.in_count for t in trackers.values())
            out_c = sum(t.out_count for t in trackers.values())
            current = in_c - out_c
            max_cap = config['max_capacity']
            warn_lim = max_cap * config['warn_threshold'] / 100
            status = 'green' if current < warn_lim else 'yellow' if current < max_cap else 'red'
            handle_status_change(status)
            await ws.send_json({
                'in_count': in_c,
                'out_count': out_c,
                'current': current,
                'max_capacity': max_cap,
                'status': status,
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info('WebSocket disconnected')
    except Exception as exc:
        logger.exception('Error in ws_stats: %s', exc)


@app.get('/settings')
async def settings_page(request: Request):
    return templates.TemplateResponse(
        'settings.html',
        {
            'request': request,
            'cfg': config,
            'ppe_items': PPE_ITEMS,
            'anomaly_items': ANOMALY_ITEMS,
            'count_options': list(COUNT_GROUPS.keys()),
        },
    )


@app.post('/settings')
async def update_settings(request: Request):
    data = await request.json()
    for key in [
        'max_capacity',
        'warn_threshold',
        'fps',
        'skip_frames',
        'line_ratio',
        'v_thresh',
        'debounce',
        'retry_interval',
        'conf_thresh',
        'reset_schedule',
        'person_model',
        'ppe_model',
    ]:
        if key in data:
            val = data[key]
            if key == 'reset_schedule':
                config[key] = val
            else:
                config[key] = type(config.get(key, val))(val)
    if 'track_ppe' in data and isinstance(data['track_ppe'], list):
        config['track_ppe'] = data['track_ppe']
    if 'alert_anomalies' in data and isinstance(data['alert_anomalies'], list):
        config['alert_anomalies'] = data['alert_anomalies']
    if 'track_objects' in data and isinstance(data['track_objects'], list):
        config['track_objects'] = data['track_objects']
    if 'line_orientation' in data:
        config['line_orientation'] = data['line_orientation']
    save_config(config, config_path, redis_client)
    for tr in trackers.values():
        tr.update_cfg(config)
    return {"saved": True}


@app.post('/reset')
async def reset_endpoint():
    reset_counts()
    redis_client.set('last_reset', time.time())
    return {'reset': True}


@app.get('/cameras')
async def cameras_page(request: Request):
    return templates.TemplateResponse(
        'cameras.html',
        {'request': request, 'cams': cameras}
    )


@app.post('/cameras')
async def add_camera(request: Request):
    """Add a new camera without validating the stream."""
    data = await request.json()
    name = data.get('name') or f"Camera{len(cameras)+1}"
    url = data.get('url')
    mode = data.get('mode', 'both')
    if not url:
        return {'error': 'Missing URL'}
    cam_id = max([c['id'] for c in cameras], default=0) + 1
    cam = {
        'id': cam_id,
        'name': name,
        'url': url,
        'mode': mode,
        'enabled': True,
    }
    cameras.append(cam)
    save_cameras(cameras, redis_client)
    start_tracker(cam)
    return {'added': True, 'camera': cam}


@app.delete('/cameras/{cam_id}')
async def delete_camera(cam_id: int):
    """Delete a camera by id."""
    global cameras
    remaining = [c for c in cameras if c['id'] != cam_id]
    if len(remaining) == len(cameras):
        return {'error': 'Not found'}
    cameras[:] = remaining
    stop_tracker(cam_id)
    save_cameras(cameras, redis_client)
    return {'deleted': True}


@app.patch('/cameras/{cam_id}')
async def toggle_camera(cam_id: int):
    for cam in cameras:
        if cam['id'] == cam_id:
            cam['enabled'] = not cam.get('enabled', True)
            if cam['enabled']:
                start_tracker(cam)
            else:
                stop_tracker(cam_id)
            save_cameras(cameras, redis_client)
            return {'enabled': cam['enabled']}
    return {'error': 'Not found'}


@app.get('/email')
async def email_page(request: Request):
    return templates.TemplateResponse('email.html', {'request': request, 'email': config.get('email', {})})


@app.post('/email')
async def update_email(request: Request):
    data = await request.json()
    config['email'].update(data)
    save_config(config, config_path, redis_client)
    return {'saved': True}


@app.get('/email/test')
async def test_email():
    try:
        send_email('Test Email', 'This is a test email from Crowd Manager', [config['email'].get('from_addr','')])
        return {'sent': True}
    except Exception as exc:
        return {'sent': False, 'error': str(exc)}


@app.get('/report')
async def report_page(request: Request):
    return templates.TemplateResponse('report.html', {'request': request})


@app.get("/report_data")
async def report_data(start: str, end: str):
    """Return logged counts for a given time range."""
    try:
        start_ts = int(datetime.fromisoformat(start).timestamp())
        end_ts = int(datetime.fromisoformat(end).timestamp())
    except Exception:
        return {"error": "invalid range"}

    entries = redis_client.lrange("history", 0, -1)
    times, ins, outs, currents = [], [], [], []
    for item in entries:
        entry = json.loads(item)
        ts = entry.get("ts")
        if ts is None or ts < start_ts or ts > end_ts:
            continue
        times.append(datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"))
        i = entry.get("in", 0)
        o = entry.get("out", 0)
        ins.append(i)
        outs.append(o)
        currents.append(i - o)

    return {"times": times, "ins": ins, "outs": outs, "current": currents}

@app.get("/report/export")
async def report_export(start: str, end: str):
    data = await report_data(start, end)
    if "error" in data:
        return JSONResponse(data, status_code=400)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Time", "In", "Out", "Current"])
    for row in zip(data["times"], data["ins"], data["outs"], data["current"]):
        writer.writerow(row)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=report.csv"},
    )
@app.on_event('startup')
async def on_startup():
    asyncio.create_task(auto_reset_loop())
    asyncio.create_task(count_log_loop())


def main():
    global config, config_path, redis_client, cameras
    parser = argparse.ArgumentParser()
    parser.add_argument('stream_url', nargs='?')
    parser.add_argument('-c', '--config', default='config.json')
    parser.add_argument('-w', '--workers', type=int, default=None)
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else str(BASE_DIR / args.config)
    temp_cfg = json.load(open(config_path))
    redis_client = redis.Redis.from_url(temp_cfg.get('redis_url', 'redis://localhost:6379/0'))
    config = load_config(config_path, redis_client)
    cameras = load_cameras(
        redis_client,
        config['stream_url'],
    )
    if args.stream_url:
        cameras = [{
            'id': 1,
            'name': 'CameraCLI',
            'url': args.stream_url,
            'mode': 'both',
            'enabled': True,
        }]
    for cam in cameras:
        if cam.get('enabled', True):
            start_tracker(cam)

    cores = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else config['default_workers']
    w = max((cores-1 if workers == -1 else (1 if workers == 0 else workers)), 1)
    cv2.setNumThreads(w)
    torch.set_num_threads(w)
    logger.info(f"Threads={w}, cores={cores}")

    logger.info(f"Server http://0.0.0.0:{config['port']}")
    uvicorn.run(app, host='0.0.0.0', port=config['port'], log_config=None)


if __name__ == '__main__':
    main()
