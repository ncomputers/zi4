#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import queue
import threading
import time
from datetime import date
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from loguru import logger
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import redis
import uvicorn

CONFIG_KEY = "config"
redis_client: redis.Redis | None = None

BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
lock = threading.Lock()
output_frame = None


def load_config(path: str, r: redis.Redis) -> dict:
    if r.exists(CONFIG_KEY):
        try:
            return json.loads(r.get(CONFIG_KEY))
        except Exception as e:
            logger.warning(f"Failed to load config from redis: {e}")
    with open(path, "r") as f:
        cfg = json.load(f)
    r.set(CONFIG_KEY, json.dumps(cfg))
    return cfg


def save_config(cfg: dict, path: str, r: redis.Redis) -> None:
    r.set(CONFIG_KEY, json.dumps(cfg))
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


class FlowTracker:
    """Track entry and exit counts using YOLOv8 and DeepSORT."""

    def __init__(self, source: str, cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.source = source
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model {self.model_path} on {self.device}")
        self.model = YOLO(self.model_path)
        if self.device.startswith("cuda"):
            self.model.model.to(self.device)
            torch.backends.cudnn.benchmark = True
        self.tracker = DeepSort(max_age=5)
        self.frame_queue = queue.Queue(maxsize=10)
        self.tracks = {}
        self.redis = redis.Redis.from_url(self.redis_url)
        self.in_count = int(self.redis.get("in_count") or 0)
        self.out_count = int(self.redis.get("out_count") or 0)
        stored_date = self.redis.get("count_date")
        self.prev_date = (
            date.fromisoformat(stored_date.decode()) if stored_date else date.today()
        )
        self.redis.mset(
            {
                "in_count": self.in_count,
                "out_count": self.out_count,
                "count_date": self.prev_date.isoformat(),
            }
        )
        self.running = True

    def update_config(self, cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)

    def capture_loop(self):
        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                logger.error(f"Cannot open stream: {self.source}")
                time.sleep(self.retry_interval)
                continue
            logger.info(f"Stream opened: {self.source}")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Lost stream, retrying in {self.retry_interval}s"
                    )
                    break
                if self.frame_queue.full():
                    _ = self.frame_queue.get()
                self.frame_queue.put(frame)
            cap.release()
            time.sleep(self.retry_interval)

    def process_loop(self):
        global output_frame
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
                self.redis.mset(
                    {
                        "in_count": self.in_count,
                        "out_count": self.out_count,
                        "count_date": self.prev_date.isoformat(),
                    }
                )
                logger.info("Daily counts reset")
            if self.skip_frames and idx % self.skip_frames:
                continue
            res = self.model.predict(frame, device=self.device, verbose=False)[0]
            h, w = frame.shape[:2]
            x_line = int(w * self.line_ratio)
            cv2.line(frame, (x_line, 0), (x_line, h), (255, 0, 0), 2)
            dets = [
                (
                    [
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2] - xyxy[0]),
                        int(xyxy[3] - xyxy[1]),
                    ],
                    conf,
                    "person",
                )
                for *xyxy, conf, cls in res.boxes.data.tolist()
                if int(cls) == 0 and conf >= self.conf_thresh
            ]
            tracks = self.tracker.update_tracks(dets, frame=frame)
            now = time.time()
            for tr in tracks:
                if not tr.is_confirmed():
                    continue
                tid = tr.track_id
                x1, y1, x2, y2 = map(int, tr.to_ltrb())
                cx = (x1 + x2) // 2
                zone = "left" if cx < x_line else "right"
                if tid not in self.tracks:
                    self.tracks[tid] = {
                        "zone": zone,
                        "cx": cx,
                        "time": now,
                        "last_counted": None,
                    }
                prev = self.tracks[tid]
                if (
                    zone != prev["zone"]
                    and abs(cx - prev["cx"]) > self.v_thresh
                    and now - prev["time"] > self.debounce
                ):
                    direction = None
                    if prev["zone"] == "left" and zone == "right":
                        direction = "Entering"
                    elif prev["zone"] == "right" and zone == "left":
                        direction = "Exiting"
                    if direction:
                        if prev["last_counted"] is None:
                            if direction == "Entering":
                                self.in_count += 1
                            else:
                                self.out_count += 1
                            self.redis.mset(
                                {
                                    "in_count": self.in_count,
                                    "out_count": self.out_count,
                                }
                            )
                            prev["last_counted"] = direction
                            logger.info(
                                f"{direction} ID{tid}: In={self.in_count} Out={self.out_count}"
                            )
                        elif prev["last_counted"] != direction:
                            if prev["last_counted"] == "Entering":
                                self.in_count -= 1
                            else:
                                self.out_count -= 1
                            self.redis.mset(
                                {
                                    "in_count": self.in_count,
                                    "out_count": self.out_count,
                                }
                            )
                            prev["last_counted"] = None
                            logger.info(f"Reversed flow for ID{tid}")
                        prev["time"] = now
                prev["zone"], prev["cx"] = zone, cx
                color = (0, 255, 0) if zone == "right" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID{tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            cv2.putText(
                frame,
                f"Entering: {self.in_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Exiting: {self.out_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            with lock:
                output_frame = frame.copy()
            time.sleep(1 / self.fps)


app = FastAPI()
counter: FlowTracker | None = None
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
config: dict
config_path: str


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "MAX_CAPACITY": counter.max_capacity,
            "WARN_THRESHOLD": counter.warn_threshold,
        },
    )


@app.get("/video_feed")
async def video_feed():
    async def gen():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    await asyncio.sleep(0.1)
                    continue
                _, buf = cv2.imencode(".jpg", output_frame)
                frame = buf.tobytes()
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            await asyncio.sleep(1 / counter.fps)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws/stats")
async def ws_stats(ws: WebSocket):
    await ws.accept()
    while True:
        in_count = counter.in_count
        out_count = counter.out_count
        current = in_count - out_count
        max_cap = counter.max_capacity
        warn_lim = max_cap * counter.warn_threshold / 100
        status = "green" if current < warn_lim else "yellow" if current < max_cap else "red"
        await ws.send_json(
            {
                "in_count": in_count,
                "out_count": out_count,
                "current": current,
                "status": status,
                "max_capacity": max_cap,
            }
        )
        await asyncio.sleep(1)


@app.get("/settings")
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request, "cfg": config})


@app.post("/settings")
async def update_settings(request: Request):
    form = await request.form()
    def parse_num(key, cast):
        if key in form:
            config[key] = cast(form[key])
            setattr(counter, key, config[key])
    parse_num("max_capacity", int)
    parse_num("warn_threshold", int)
    parse_num("fps", int)
    parse_num("skip_frames", int)
    parse_num("line_ratio", float)
    parse_num("v_thresh", int)
    parse_num("debounce", float)
    parse_num("retry_interval", int)
    parse_num("conf_thresh", float)
    counter.update_config(config)
    save_config(config, config_path, redis_client)
    return templates.TemplateResponse("settings.html", {"request": request, "cfg": config, "saved": True})


def main():
    global counter, config, config_path, redis_client

    parser = argparse.ArgumentParser()
    parser.add_argument("stream_url", nargs="?")
    parser.add_argument("-c", "--config", default="config.json")
    parser.add_argument("-w", "--workers", type=int, default=None)
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else str(BASE_DIR / args.config)
    # initialize redis and load configuration
    temp_cfg = json.load(open(config_path))
    redis_client = redis.Redis.from_url(temp_cfg.get("redis_url", "redis://localhost:6379/0"))
    config = load_config(config_path, redis_client)
    url = args.stream_url or config["stream_url"]

    cores = os.cpu_count() or 1
    workers = args.workers if args.workers is not None else config["default_workers"]
    w = max((cores - 1 if workers == -1 else (1 if workers == 0 else workers)), 1)
    cv2.setNumThreads(w)
    torch.set_num_threads(w)
    logger.info(f"Threads={w}, cores={cores}")

    counter = FlowTracker(url, config)
    threading.Thread(target=counter.capture_loop, daemon=True).start()
    threading.Thread(target=counter.process_loop, daemon=True).start()

    logger.info(f"Server http://0.0.0.0:{config['port']} Stream={url}")
    uvicorn.run(app, host="0.0.0.0", port=config["port"], log_config=None)


if __name__ == "__main__":
    main()
