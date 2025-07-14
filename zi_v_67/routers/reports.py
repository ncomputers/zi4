"""Count report routes."""
from __future__ import annotations
from typing import Dict
from datetime import datetime
import io
import csv
import json

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from modules.utils import require_roles

router = APIRouter()

def init_context(config: dict, trackers: Dict[int, "FlowTracker"], redis_client, templates_path):
    global cfg, trackers_map, redis, templates
    cfg = config
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)

@router.get('/report')
async def report_page(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    return templates.TemplateResponse('report.html', {'request': request})

@router.get('/report_data')
async def report_data(start: str, end: str, request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    try:
        start_ts = int(datetime.fromisoformat(start).timestamp())
        end_ts = int(datetime.fromisoformat(end).timestamp())
    except Exception:
        return {"error": "invalid range"}

    entries = redis.lrange('history', 0, -1)
    times, ins, outs, currents = [], [], [], []
    for item in entries:
        entry = json.loads(item)
        ts = entry.get('ts')
        if ts is None or ts < start_ts or ts > end_ts:
            continue
        times.append(datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'))
        i = entry.get('in', 0)
        o = entry.get('out', 0)
        ins.append(i)
        outs.append(o)
        currents.append(i - o)
    return {'times': times, 'ins': ins, 'outs': outs, 'current': currents}

@router.get('/report/export')
async def report_export(start: str, end: str, request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await report_data(start, end, request)
    if 'error' in data:
        return JSONResponse(data, status_code=400)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Time', 'In', 'Out', 'Current'])
    for row in zip(data['times'], data['ins'], data['outs'], data['current']):
        writer.writerow(row)
    return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=report.csv'})
