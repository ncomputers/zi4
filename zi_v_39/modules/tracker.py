from __future__ import annotations
import queue
import threading
import time
import json
from datetime import date, datetime
import cv2
import torch
from loguru import logger
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import redis
from pathlib import Path
from .utils import send_email, lock, SNAP_DIR

class FlowTracker:
    """Tracks entry and exit counts using YOLOv8 and DeepSORT."""

    def __init__(self, cam_id: int, src: str, classes: list[str], cfg: dict):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.cam_id = cam_id
        self.src = src
        self.classes = classes
        self.count_classes = cfg.get("count_classes", [])
        self.ppe_classes = cfg.get("ppe_classes", [])
        self.alert_anomalies = cfg.get("alert_anomalies", [])
        self.line_orientation = cfg.get("line_orientation", "vertical")
        self.helmet_conf_thresh = cfg.get("helmet_conf_thresh", 0.5)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading person model {self.person_model} on {self.device}")
        self.model_person = YOLO(self.person_model)
        logger.info(f"Loading PPE model {self.ppe_model} on {self.device}")
        self.model_ppe = YOLO(self.ppe_model)
        self.email_cfg = cfg.get("email", {})
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
        self.snap_dir = SNAP_DIR
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
        if "ppe_classes" in cfg:
            self.ppe_classes = cfg["ppe_classes"]
        if "alert_anomalies" in cfg:
            self.alert_anomalies = cfg["alert_anomalies"]
        if "line_orientation" in cfg:
            self.line_orientation = cfg["line_orientation"]
        if "helmet_conf_thresh" in cfg:
            self.helmet_conf_thresh = cfg["helmet_conf_thresh"]
        if "person_model" in cfg and cfg["person_model"] != getattr(self, "person_model", None):
            self.person_model = cfg["person_model"]
            self.model_person = YOLO(self.person_model)
            if self.device.startswith("cuda"):
                self.model_person.model.to(self.device)
        if "email" in cfg:
            self.email_cfg = cfg["email"]
        if "ppe_model" in cfg and cfg["ppe_model"] != getattr(self, "ppe_model", None):
            self.ppe_model = cfg["ppe_model"]
            self.model_ppe = YOLO(self.ppe_model)
            if self.device.startswith("cuda"):
                self.model_ppe.model.to(self.device)

    def check_helmet(self, img):
        """Return ('helmet' or 'no_helmet', confidence)."""
        res = self.model_ppe.predict(img, device=self.device, verbose=False)[0]
        helmet_boxes = []
        head_boxes = []
        for *xyxy, conf, cls in res.boxes.data.tolist():
            label = (
                self.model_ppe.names[int(cls)]
                if isinstance(self.model_ppe.names, dict)
                else self.model_ppe.names[int(cls)]
            ).lower()
            x1, y1, x2, y2 = map(int, xyxy)
            if label == "helmet":
                helmet_boxes.append((x1, y1, x2, y2, conf))
            elif label == "head":
                head_boxes.append((x1, y1, x2, y2, conf))
        status = None
        best = 0.0
        for x1, y1, x2, y2, conf in head_boxes:
            covered = False
            for hx1, hy1, hx2, hy2, _ in helmet_boxes:
                if not (hx2 < x1 or hx1 > x2 or hy2 < y1 or hy1 > y2):
                    covered = True
                    break
            if not covered and conf > best:
                status = "no_helmet"
                best = conf
        if status is None and helmet_boxes:
            status = "helmet"
            best = max(h[-1] for h in helmet_boxes)
        return status, best

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
                        'first_zone': zone,
                    }
                prev = self.tracks[tid]
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
                first_zone = info.get('first_zone')
                last_zone = info.get('zone')
                img = info.get('best_img')
                status = None
                best_conf = 0.0
                if img is not None and img.size:
                    status, best_conf = self.check_helmet(img)
                    logger.info(f"Helmet check ID{tid}: {status or 'unknown'} conf={best_conf:.2f}")
                # fallback count using ROI zones when no crossing detected
                if info.get('last') is None and first_zone and last_zone and first_zone != last_zone and info.get('label') in self.count_classes:
                    direction = None
                    if self.line_orientation == 'horizontal':
                        if first_zone == 'top' and last_zone == 'bottom':
                            direction = 'Entering'
                        elif first_zone == 'bottom' and last_zone == 'top':
                            direction = 'Exiting'
                    else:
                        if first_zone == 'left' and last_zone == 'right':
                            direction = 'Entering'
                        elif first_zone == 'right' and last_zone == 'left':
                            direction = 'Exiting'
                    if direction:
                        if direction == 'Entering':
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        self.redis.mset({self.key_in: self.in_count, self.key_out: self.out_count})
                        logger.info(f"ROI {direction} ID{tid}: In={self.in_count} Out={self.out_count}")
                if status and best_conf >= self.helmet_conf_thresh:
                    ts = int(time.time())
                    snap = img if img is not None else frame
                    fname = f'{self.cam_id}_{tid}_{status}_{int(best_conf*100)}_{ts}.jpg'
                    path = self.snap_dir / fname
                    cv2.imwrite(str(path), snap)
                    entry = {
                        'ts': ts,
                        'cam_id': self.cam_id,
                        'track_id': tid,
                        'status': status,
                        'conf': best_conf,
                        'path': str(path),
                    }
                    self.redis.rpush('helmet_logs', json.dumps(entry))
                    if status == 'no_helmet' and 'no_helmet' in self.alert_anomalies:
                        to_emails = self.email_cfg.get('ppe_to', '')
                        recipients = [a.strip() for a in to_emails.split(',') if a.strip()]
                        threading.Thread(
                            target=send_email,
                            args=(f'PPE Alert: {status}', f'Camera {self.cam_id} detected {status}', recipients, open(path, 'rb').read(), self.email_cfg),
                            daemon=True,
                        ).start()

            cv2.putText(frame, f"Entering: {self.in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exiting: {self.out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with lock:
                self.output_frame = frame.copy()
            time.sleep(1 / self.fps)


