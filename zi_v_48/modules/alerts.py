from __future__ import annotations
import json
import io
import time
import threading
from loguru import logger
from .utils import send_email
import redis
from datetime import datetime, timedelta
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from pathlib import Path

class AlertWorker:
    def __init__(self, cfg: dict, redis_url: str, base_dir: Path):
        self.cfg = cfg
        self.redis = redis.Redis.from_url(redis_url)
        self.base_dir = base_dir
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)

    def loop(self):
        while self.running:
            try:
                self.check_rules()
            except Exception as exc:
                logger.error("alert loop error: %s", exc)
            time.sleep(60)

    def _collect_rows(self, start_ts: int, end_ts: int, metric: str):
        entries = self.redis.lrange('ppe_logs', 0, -1)
        rows = []
        count = 0
        for item in entries:
            e = json.loads(item)
            ts = e.get('ts')
            if ts is None or ts < start_ts or ts > end_ts:
                continue
            if e.get('status') == metric:
                count += 1
            rows.append(e)
        return count, rows

    def _send_report(self, rows, recipients, subject):
        wb = Workbook()
        ws = wb.active
        ws.append(['Time','Camera','Track','Status','Conf','Color'])
        for r in rows:
            ws.append([
                datetime.fromtimestamp(r['ts']).strftime('%Y-%m-%d %H:%M'),
                r.get('cam_id'),
                r.get('track_id'),
                r.get('status'),
                round(r.get('conf',0),2),
                r.get('color') or ''
            ])
            path = r.get('path')
            if path and Path(path).exists():
                img = XLImage(path)
                img.width = 80
                img.height = 60
                ws.add_image(img, f'F{ws.max_row}')
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        send_email(subject, 'See attached report', recipients, None,
                   self.cfg.get('email', {}),
                   attachment=bio.getvalue(),
                   attachment_name='report.xlsx',
                   attachment_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    def check_rules(self):
        rules = self.cfg.get('alert_rules', [])
        now = time.time()
        for i, rule in enumerate(rules):
            metric = rule.get('metric')
            threshold = rule.get('threshold', 0)
            freq = rule.get('frequency', 'daily')
            recipients = [a.strip() for a in rule.get('recipients','').split(',') if a.strip()]
            if not recipients:
                continue
            last_key = f'alert_rule_{i}_last'
            last_ts = float(self.redis.get(last_key) or 0)
            interval = 3600 if freq == 'hourly' else 86400
            if now - last_ts < interval:
                continue
            count, rows = self._collect_rows(last_ts, int(now), metric)
            if count >= threshold:
                self._send_report(rows, recipients, f'Alert: {metric}')
                self.redis.set(last_key, int(now))
