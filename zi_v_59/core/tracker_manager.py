"""Manage FlowTracker instances and related counters."""
from __future__ import annotations
import json
import time
import threading
from datetime import date
import asyncio
from typing import Dict, List

import redis
from loguru import logger

from .config import load_config, save_config
from modules.tracker import FlowTracker

lock = threading.Lock()


def load_cameras(r: redis.Redis, default_url: str) -> List[dict]:
    data = r.get("cameras")
    if data:
        try:
            cams = json.loads(data)
            for cam in cams:
                cam.setdefault("tasks", ["both"])
                if cam.get("mode") and not any(t in ("in", "out", "both") for t in cam["tasks"]):
                    cam["tasks"].insert(0, cam["mode"])
                cam.pop("mode", None)
            if len(cams) == 1 and cams[0].get("id") == 1 and cams[0]["url"] != default_url:
                cams[0]["url"] = default_url
                r.set("cameras", json.dumps(cams))
            return cams
        except json.JSONDecodeError:
            pass
    cams = [
        {"id": 1, "name": "Camera1", "url": default_url, "tasks": ["both"], "enabled": True}
    ]
    r.set("cameras", json.dumps(cams))
    return cams


def save_cameras(cams: List[dict], r: redis.Redis) -> None:
    r.set("cameras", json.dumps(cams))


def start_tracker(cam: dict, cfg: dict, trackers: Dict[int, FlowTracker]) -> FlowTracker:
    tasks = cam.get("tasks", [])
    mode = "both"
    for m in ("in", "out", "both"):
        if m in tasks:
            mode = m
            tasks = [t for t in tasks if t not in ("in", "out", "both")]
            break
    tr = FlowTracker(
        cam["id"],
        cam["url"],
        cfg.get("object_classes", ["person"]),
        cfg,
        tasks,
        mode,
    )
    trackers[cam["id"]] = tr
    threading.Thread(target=tr.capture_loop, daemon=True).start()
    threading.Thread(target=tr.process_loop, daemon=True).start()
    return tr


def stop_tracker(cam_id: int, trackers: Dict[int, FlowTracker]) -> None:
    tr = trackers.pop(cam_id, None)
    if tr:
        tr.running = False


def reset_counts(trackers: Dict[int, FlowTracker]) -> None:
    for tr in trackers.values():
        tr.in_count = 0
        tr.out_count = 0
        tr.tracks.clear()
        tr.prev_date = date.today()
        tr.redis.mset({tr.key_in: 0, tr.key_out: 0, tr.key_date: tr.prev_date.isoformat()})
    logger.info("Counts reset")


def reset_nohelmet(r: redis.Redis) -> None:
    r.set("nohelmet_count", 0)
    logger.info("No-helmet counter reset")


def log_counts(r: redis.Redis, trackers: Dict[int, FlowTracker]) -> None:
    ts = int(time.time())
    in_c = sum(t.in_count for t in trackers.values())
    out_c = sum(t.out_count for t in trackers.values())
    entry = json.dumps({"ts": ts, "in": in_c, "out": out_c})
    r.rpush("history", entry)


async def count_log_loop(r: redis.Redis, trackers: Dict[int, FlowTracker]):
    while True:
        log_counts(r, trackers)
        await asyncio.sleep(60)


last_status: str | None = None

def handle_status_change(status: str, r: redis.Redis) -> None:
    global last_status
    if status == last_status:
        return
    last_status = status
    ts = int(time.time())
    if status == "yellow":
        r.incr("yellow_alert_count")
        entry = {"ts": ts, "cam_id": 0, "track_id": 0, "status": "yellow_alert", "conf": 0, "color": None, "path": None}
        r.rpush("ppe_logs", json.dumps(entry))
    elif status == "red":
        r.incr("red_alert_count")
        entry = {"ts": ts, "cam_id": 0, "track_id": 0, "status": "red_alert", "conf": 0, "color": None, "path": None}
        r.rpush("ppe_logs", json.dumps(entry))
