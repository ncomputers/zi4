"""Email and alert rule management routes."""
from __future__ import annotations
from typing import Dict

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..modules.utils import send_email
from ..core.config import ANOMALY_ITEMS, save_config

router = APIRouter()

def init_context(config: dict, trackers: Dict[int, "FlowTracker"], redis_client, templates_path, config_path: str):
    global cfg, trackers_map, redis, templates, cfg_path
    cfg = config
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)
    cfg_path = config_path

@router.get('/alerts')
async def alerts_page(request: Request):
    return templates.TemplateResponse('email_alerts.html', {
        'request': request,
        'rules': cfg.get('alert_rules', []),
        'email': cfg.get('email', {}),
        'anomaly_items': ANOMALY_ITEMS,
    })

@router.post('/alerts')
async def save_alerts(request: Request):
    data = await request.json()
    cfg['alert_rules'] = data.get('rules', [])
    save_config(cfg, cfg_path, redis)
    return {'saved': True}

@router.post('/email')
async def update_email(request: Request):
    data = await request.json()
    cfg['email'].update(data)
    save_config(cfg, cfg_path, redis)
    return {'saved': True}

@router.get('/email/test')
async def test_email():
    try:
        send_email('Test Email', 'This is a test email from Crowd Manager', [cfg['email'].get('from_addr','')], cfg=cfg.get('email', {}))
        return {'sent': True}
    except Exception as exc:
        return {'sent': False, 'error': str(exc)}
