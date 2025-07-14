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
from .duplicate_filter import DuplicateFilter
from core.config import ANOMALY_ITEMS

class FlowTracker:
    """Tracks entry and exit counts using YOLOv8 and DeepSORT."""

    def __init__(self, cam_id: int, src: str, classes: list[str], cfg: dict,
                 tasks: list[str] | None = None, mode: str = "both",
                 src_type: str = "http", line_orientation: str | None = None,
                 reverse: bool = False, resolution: str = "original"):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.cam_id = cam_id
        self.src = src
        self.src_type = src_type
        self.classes = classes
        self.tasks = tasks or []
        self.mode = mode
        self.count_classes = cfg.get("count_classes", [])
        self.ppe_classes = cfg.get("ppe_classes", [])
        self.alert_anomalies = cfg.get("alert_anomalies", [])
        self.line_orientation = line_orientation or cfg.get("line_orientation", "vertical")
        self.reverse = reverse
        self.resolution = resolution
        self.helmet_conf_thresh = cfg.get("helmet_conf_thresh", 0.5)
        self.detect_helmet_color = cfg.get("detect_helmet_color", False)
        self.track_misc = cfg.get("track_misc", True)
        self.show_lines = cfg.get("show_lines", True)
        self.show_ids = cfg.get("show_ids", True)
        self.show_track_lines = cfg.get("show_track_lines", False)
        self.duplicate_filter_enabled = cfg.get("duplicate_filter_enabled", False)
        self.duplicate_filter_threshold = cfg.get("duplicate_filter_threshold", 0.1)
        self.duplicate_bypass_seconds = cfg.get("duplicate_bypass_seconds", 2)
        self.max_retry = cfg.get("max_retry", 5)
        self.online = False
        
        self.dup_filter = DuplicateFilter(self.duplicate_filter_threshold, self.duplicate_bypass_seconds) if self.duplicate_filter_enabled else None
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        self.device = cfg.get("device")
        if not self.device or self.device == "auto":
            self.device = "cuda:0" if cuda_available else "cpu"
        if self.device.startswith("cuda") and not cuda_available:
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        logger.info(f"Loading person model {self.person_model} on {self.device}")
        self.model_person = YOLO(self.person_model)
        logger.info(f"Loading PPE model {self.ppe_model} on {self.device}")
        self.model_ppe = YOLO(self.ppe_model)
        self.email_cfg = cfg.get("email", {})
        if self.device.startswith("cuda"):
            self.model_person.model.to(self.device).half()
            self.model_ppe.model.to(self.device).half()
            torch.backends.cudnn.benchmark = True
        else:
            self.model_person.model.to(self.device)
            self.model_ppe.model.to(self.device)
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
        today = date.today().isoformat()
        for item in ANOMALY_ITEMS:
            date_key = f'{item}_date'
            count_key = f'{item}_count'
            d_raw = self.redis.get(date_key)
            d = date.fromisoformat(d_raw.decode()) if d_raw else self.prev_date
            if d.isoformat() != today:
                self.redis.mset({count_key: 0, date_key: today})
        self.snap_dir = SNAP_DIR
        self.output_frame = None
        self.running = True

    @staticmethod
    def _clean_label(name: str) -> str:
        """Normalize a label to lowercase with underscores."""
        return name.lower().replace(' ', '_').replace('-', '_').replace('/', '_')

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
        if "tasks" in cfg:
            self.tasks = cfg["tasks"]
        if "mode" in cfg:
            self.mode = cfg["mode"]
        if "type" in cfg:
            self.src_type = cfg["type"]
        if "alert_anomalies" in cfg:
            self.alert_anomalies = cfg["alert_anomalies"]
        if "line_orientation" in cfg:
            self.line_orientation = cfg["line_orientation"]
        if "reverse" in cfg:
            self.reverse = bool(cfg["reverse"])
        if "resolution" in cfg:
            self.resolution = cfg["resolution"]
        if "helmet_conf_thresh" in cfg:
            self.helmet_conf_thresh = cfg["helmet_conf_thresh"]
        if "detect_helmet_color" in cfg:
            self.detect_helmet_color = cfg["detect_helmet_color"]
        if "track_misc" in cfg:
            self.track_misc = cfg["track_misc"]
        if "show_lines" in cfg:
            self.show_lines = cfg["show_lines"]
        if "show_ids" in cfg:
            self.show_ids = cfg["show_ids"]
        if "show_track_lines" in cfg:
            self.show_track_lines = cfg["show_track_lines"]
        if "duplicate_filter_enabled" in cfg:
            self.duplicate_filter_enabled = cfg["duplicate_filter_enabled"]
            self.dup_filter = DuplicateFilter(self.duplicate_filter_threshold, self.duplicate_bypass_seconds) if self.duplicate_filter_enabled else None
        if "duplicate_filter_threshold" in cfg:
            self.duplicate_filter_threshold = cfg["duplicate_filter_threshold"]
            if self.dup_filter:
                self.dup_filter.threshold = self.duplicate_filter_threshold
        if "duplicate_bypass_seconds" in cfg:
            self.duplicate_bypass_seconds = cfg["duplicate_bypass_seconds"]
            if self.dup_filter:
                self.dup_filter.bypass_seconds = self.duplicate_bypass_seconds
        if "person_model" in cfg and cfg["person_model"] != getattr(self, "person_model", None):
            self.person_model = cfg["person_model"]
            self.model_person = YOLO(self.person_model)
            if self.device.startswith("cuda"):
                self.model_person.model.to(self.device).half()
        if "email" in cfg:
            self.email_cfg = cfg["email"]
        if "ppe_model" in cfg and cfg["ppe_model"] != getattr(self, "ppe_model", None):
            self.ppe_model = cfg["ppe_model"]
            self.model_ppe = YOLO(self.ppe_model)
            if self.device.startswith("cuda"):
                self.model_ppe.model.to(self.device).half()

    def _color_name(self, bgr):
        import numpy as np
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv
        if v < 50:
            return 'black'
        if s < 40:
            return 'white'
        if h < 15 or h >= 165:
            return 'red'
        if h < 45:
            return 'yellow'
        if h < 85:
            return 'green'
        if h < 125:
            return 'blue'
        if h < 165:
            return 'purple'
        return 'unknown'

    def check_helmet(self, img):
        """Return ('helmet' or 'no_helmet', confidence, color)."""
        res = self.model_ppe.predict(img, device=self.device, verbose=False)[0]
        helmet_boxes = []
        head_boxes = []
        for *xyxy, conf, cls in res.boxes.data.tolist():
            raw = (
                self.model_ppe.names[int(cls)]
                if isinstance(self.model_ppe.names, dict)
                else self.model_ppe.names[int(cls)]
            )
            label = self._clean_label(raw)
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
        color = None
        if status is None and helmet_boxes:
            status = "helmet"
            best_box = max(helmet_boxes, key=lambda b: b[-1])
            best = best_box[-1]
            if self.detect_helmet_color:
                hx1, hy1, hx2, hy2, _ = best_box
                crop = img[hy1:hy2, hx1:hx2]
                if crop.size:
                    bgr = tuple(int(x) for x in crop.mean(axis=(0,1)))
                    color = self._color_name(bgr)
        return status, best, color

    def check_ppe(self, img) -> dict:
        """Return detection results for configured tasks."""
        res = self.model_ppe.predict(img, device=self.device, verbose=False)[0]
        results = {}
        scores = {}
        boxes = {}
        for *xyxy, conf, cls in res.boxes.data.tolist():
            raw = (
                self.model_ppe.names[int(cls)]
                if isinstance(self.model_ppe.names, dict)
                else self.model_ppe.names[int(cls)]
            )
            label = self._clean_label(raw)
            if conf > scores.get(label, 0):
                scores[label] = conf
                boxes[label] = [int(v) for v in xyxy]
        # helmet special case
        if 'helmet' in self.tasks:
            helmet_conf = scores.get('helmet', 0)
            head_conf = scores.get('head', 0)
            color = None
            if helmet_conf and (helmet_conf >= head_conf):
                status = 'helmet'
                conf = helmet_conf
                if self.detect_helmet_color and 'helmet' in boxes:
                    hx1, hy1, hx2, hy2 = boxes['helmet']
                    crop = img[hy1:hy2, hx1:hx2]
                    if crop.size:
                        bgr = tuple(int(x) for x in crop.mean(axis=(0,1)))
                        color = self._color_name(bgr)
            else:
                status = 'no_helmet'
                conf = head_conf
            results['helmet'] = (status, conf, color)
        for item in self.tasks:
            if item == 'helmet':
                continue
            conf = scores.get(item, 0)
            if conf >= self.helmet_conf_thresh:
                results[item] = (item, conf, None)
            else:
                results[item] = (f'no_{item}', conf, None)
        return results

    def _open_capture(self):
        if self.src_type == "rtsp":
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.src_type == "local":
            try:
                index = int(self.src)
            except ValueError:
                index = self.src
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(self.src)
        if self.resolution != "original":
            res_map = {
                "480p": (640, 480),
                "720p": (1280, 720),
                "1080p": (1920, 1080),
            }
            if self.resolution in res_map:
                w, h = res_map[self.resolution]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    def capture_loop(self):
        failures = 0
        while self.running:
            try:
                cap = self._open_capture()
                if not cap.isOpened():
                    raise RuntimeError("open_failed")
                self.online = True
                failures = 0
                logger.info(f"Stream opened: {self.src}")
                while self.running:
                    try:
                        ret, frame = cap.read()
                    except (ConnectionResetError, BrokenPipeError, TimeoutError) as e:
                        logger.error(f"Stream read error: {e}")
                        ret = False
                    if not ret:
                        logger.warning(f"Lost stream, retry in {self.retry_interval}s")
                        failures += 1
                        if failures >= self.max_retry:
                            logger.error(f"Max retries reached for {self.src}. stopping tracker")
                            self.running = False
                            break
                        break
                    if self.frame_queue.full():
                        _ = self.frame_queue.get()
                    self.frame_queue.put(frame)
                cap.release()
            except Exception as e:
                self.online = False
                logger.error(f"Cannot open stream: {self.src} ({e})")
                failures += 1
                if failures >= self.max_retry:
                    logger.error(f"Max retries reached for {self.src}. stopping tracker")
                    self.running = False
            if self.running:
                time.sleep(self.retry_interval)

    def process_loop(self):
        idx = 0
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            if self.dup_filter and self.dup_filter.is_duplicate(frame):
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
                for item in ANOMALY_ITEMS:
                    self.redis.mset({f'{item}_count': 0, f'{item}_date': self.prev_date.isoformat()})
                logger.info("Daily counts reset")
            if self.skip_frames and idx % self.skip_frames:
                continue
            res = self.model_person.predict(frame, device=self.device, verbose=False)[0]
            h, w = frame.shape[:2]
            if self.line_orientation == 'horizontal':
                line_pos = int(h * self.line_ratio)
                if self.show_lines:
                    cv2.line(frame, (0, line_pos), (w, line_pos), (255, 0, 0), 2)
            else:
                line_pos = int(w * self.line_ratio)
                if self.show_lines:
                    cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 0, 0), 2)
            dets = []
            for *xyxy, conf, cls in res.boxes.data.tolist():
                raw = self.model_person.names[int(cls)] if isinstance(self.model_person.names, dict) else self.model_person.names[int(cls)]
                label = self._clean_label(raw)
                if label in self.classes and conf >= self.conf_thresh:
                    dets.append([
                        [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                        conf,
                        label,
                    ])
            try:
                tracks = self.tracker.update_tracks(dets, frame=frame)
            except ValueError as e:
                logger.warning(f"tracker update error: {e}")
                continue
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
                        'images': [],
                        'first_zone': zone,
                        'trail': [(cx, cy)],
                    }
                prev = self.tracks[tid]
                conf = getattr(tr, 'det_conf', 0) or 0
                if label == 'person' and conf > prev.get('best_conf', 0):
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        prev['best_conf'] = conf
                        prev['best_img'] = crop.copy()
                if label == 'person' and conf >= 0.5:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        imgs = prev.setdefault('images', [])
                        if len(imgs) < 20:
                            imgs.append((conf, crop.copy()))
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
                    if self.reverse and direction:
                        direction = 'Exiting' if direction == 'Entering' else 'Entering'
                    if direction and prev.get('label') in self.count_classes:
                        if self.mode == 'in' and direction != 'Entering':
                            direction = None
                        elif self.mode == 'out' and direction != 'Exiting':
                            direction = None
                        if direction:
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
                trail = prev.setdefault('trail', [])
                trail.append((cx, cy))
                if len(trail) > 20:
                    trail.pop(0)
                color = (0, 255, 0) if zone == 'right' else (0, 0, 255)
                if self.show_track_lines:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if self.show_track_lines and len(trail) > 1:
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i-1], trail[i], (0,0,255), 2)
                if self.show_ids:
                    cv2.putText(frame, f"ID{tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # process tracks that have disappeared
            gone_ids = [tid for tid in list(self.tracks.keys()) if tid not in active_ids]
            for tid in gone_ids:
                info = self.tracks.pop(tid)
                first_zone = info.get('first_zone')
                last_zone = info.get('zone')
                images = []
                best_img = info.get('best_img')
                if best_img is not None and best_img.size:
                    images.append(best_img)
                import random
                candidates = [img for c, img in info.get('images', []) if c >= 0.5 and img is not best_img]
                random.shuffle(candidates)
                images.extend(candidates[:4])
                results = {}
                if self.tasks:
                    img_for_ppe = best_img if best_img is not None else frame
                    results = self.check_ppe(img_for_ppe)
                    for item, (st, conf, col) in results.items():
                        logger.info(
                            f"PPE check {item} ID{tid}: {st} conf={conf:.2f}"
                        )
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
                    if self.reverse and direction:
                        direction = 'Exiting' if direction == 'Entering' else 'Entering'
                    if direction:
                        if self.mode == 'in' and direction != 'Entering':
                            direction = None
                        elif self.mode == 'out' and direction != 'Exiting':
                            direction = None
                    if direction:
                        if direction == 'Entering':
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        self.redis.mset({self.key_in: self.in_count, self.key_out: self.out_count})
                        logger.info(f"ROI {direction} ID{tid}: In={self.in_count} Out={self.out_count}")
                if self.track_misc and self.tasks and not results:
                    ts = int(time.time())
                    snap = best_img if best_img is not None else frame
                    fname = f"{self.cam_id}_{tid}_misc_{ts}.jpg"
                    path = self.snap_dir / fname
                    cv2.imwrite(str(path), snap)
                    entry = {
                        'ts': ts,
                        'cam_id': self.cam_id,
                        'track_id': tid,
                        'status': 'misc',
                        'conf': 0,
                        'color': None,
                        'path': str(path),
                    }
                    self.redis.rpush('ppe_logs', json.dumps(entry))
                for item, res in results.items():
                    st, conf, col = res
                    if not st or conf < self.helmet_conf_thresh:
                        continue
                    ts = int(time.time())
                    snap = best_img if best_img is not None else frame
                    fname = f"{self.cam_id}_{tid}_{st}_{int(conf*100)}_{ts}.jpg"
                    path = self.snap_dir / fname
                    cv2.imwrite(str(path), snap)
                    entry = {
                        'ts': ts,
                        'cam_id': self.cam_id,
                        'track_id': tid,
                        'status': st,
                        'conf': conf,
                        'color': col,
                        'path': str(path),
                    }
                    self.redis.rpush('ppe_logs', json.dumps(entry))
                    if st.startswith('no_'):
                        self.redis.incr(f'{st}_count')


            cv2.putText(frame, f"Entering: {self.in_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Exiting: {self.out_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            with lock:
                self.output_frame = frame.copy()
            time.sleep(1 / self.fps)


