"""Settings management routes."""
from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from core.config import PPE_ITEMS, ANOMALY_ITEMS, COUNT_GROUPS, save_config
from modules.utils import require_roles
from core.tracker_manager import reset_counts, reset_nohelmet

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
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    return templates.TemplateResponse('settings.html', {
        'request': request,
        'cfg': cfg,
        'ppe_items': PPE_ITEMS,
        'anomaly_items': ANOMALY_ITEMS,
        'count_options': list(COUNT_GROUPS.keys()),
    })

@router.post('/settings')
async def update_settings(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    if data.get('password') != cfg.get('settings_password'):
        return {'saved': False, 'error': 'auth'}
    for key in [
        'max_capacity','warn_threshold','fps','skip_frames','line_ratio','v_thresh','debounce','retry_interval','conf_thresh','helmet_conf_thresh','detect_helmet_color','show_lines','show_ids','show_track_lines','person_model','ppe_model','email_enabled','duplicate_filter_enabled','duplicate_filter_threshold','duplicate_bypass_seconds','max_retry']:
        if key in data:
            val = data[key]
            if key in ['detect_helmet_color','show_lines','show_ids','show_track_lines','email_enabled','duplicate_filter_enabled']:
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
    save_config(cfg, cfg_path, redis)
    for tr in trackers_map.values():
        tr.update_cfg(cfg)
    return {'saved': True}

@router.get('/settings/export')
async def export_settings(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    from fastapi.responses import JSONResponse
    return JSONResponse(cfg)

@router.post('/settings/import')
async def import_settings(request: Request):
    res = require_roles(request, ['admin'])
    if isinstance(res, RedirectResponse):
        return res
    data = await request.json()
    cfg.update(data)
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
