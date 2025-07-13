"""Settings management routes."""
from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..core.config import PPE_ITEMS, ANOMALY_ITEMS, COUNT_GROUPS, save_config
from ..core.tracker_manager import reset_counts, reset_nohelmet

router = APIRouter()

def init_context(config: dict, trackers: Dict[int, "FlowTracker"], redis_client, templates_path, config_path: str):
    global cfg, trackers_map, redis, templates, cfg_path
    cfg = config
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)
    cfg_path = config_path

@router.get('/settings')
async def settings_page(request: Request):
    return templates.TemplateResponse('settings.html', {
        'request': request,
        'cfg': cfg,
        'ppe_items': PPE_ITEMS,
        'anomaly_items': ANOMALY_ITEMS,
        'count_options': list(COUNT_GROUPS.keys()),
    })

@router.post('/settings')
async def update_settings(request: Request):
    data = await request.json()
    for key in [
        'max_capacity','warn_threshold','fps','skip_frames','line_ratio','v_thresh','debounce','retry_interval','conf_thresh','helmet_conf_thresh','detect_helmet_color','show_lines','show_ids','show_track_lines','person_model','ppe_model','email_enabled']:
        if key in data:
            val = data[key]
            if key in ['detect_helmet_color','show_lines','show_ids','show_track_lines','email_enabled']:
                cfg[key] = bool(val) if isinstance(val, bool) else str(val).lower() == 'true'
            else:
                cfg[key] = type(cfg.get(key, val))(val)
    if 'track_ppe' in data and isinstance(data['track_ppe'], list):
        cfg['track_ppe'] = data['track_ppe']
    if 'alert_anomalies' in data and isinstance(data['alert_anomalies'], list):
        cfg['alert_anomalies'] = data['alert_anomalies']
    if 'preview_anomalies' in data and isinstance(data['preview_anomalies'], list):
        cfg['preview_anomalies'] = data['preview_anomalies']
    if 'track_objects' in data and isinstance(data['track_objects'], list):
        cfg['track_objects'] = data['track_objects']
    if 'line_orientation' in data:
        cfg['line_orientation'] = data['line_orientation']
    save_config(cfg, cfg_path, redis)
    for tr in trackers_map.values():
        tr.update_cfg(cfg)
    return {'saved': True}

@router.post('/reset')
async def reset_endpoint():
    reset_counts(trackers_map)
    return {'reset': True}

@router.post('/reset_nohelmet')
async def reset_nohelmet_endpoint():
    reset_nohelmet(redis)
    return {'reset': True}
