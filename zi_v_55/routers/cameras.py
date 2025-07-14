"""Camera management routes."""
from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from ..core.tracker_manager import start_tracker, stop_tracker, save_cameras
from ..core.config import CAMERA_TASKS

router = APIRouter()

def init_context(config: dict, cameras: List[dict], trackers: Dict[int, "FlowTracker"], redis_client, templates_path):
    global cfg, cams, trackers_map, redis, templates
    cfg = config
    cams = cameras
    trackers_map = trackers
    redis = redis_client
    templates = Jinja2Templates(directory=templates_path)

@router.get('/cameras')
async def cameras_page(request: Request):
    return templates.TemplateResponse('cameras.html', {
        'request': request,
        'cams': cams,
        'model_classes': CAMERA_TASKS,
    })

@router.post('/cameras')
async def add_camera(request: Request):
    data = await request.json()
    name = data.get('name') or f"Camera{len(cams)+1}"
    url = data.get('url')
    tasks = data.get('tasks', [])
    if not isinstance(tasks, list):
        tasks = []
    mode = 'both'
    for m in ('in','out','both'):
        if m in tasks:
            mode = m
            tasks = [t for t in tasks if t not in ('in','out','both')]
            break
    if not url:
        return {'error': 'Missing URL'}
    cam_id = max([c['id'] for c in cams], default=0) + 1
    cam = {'id': cam_id, 'name': name, 'url': url, 'tasks': tasks, 'mode': mode, 'enabled': True}
    cams.append(cam)
    save_cameras(cams, redis)
    start_tracker(cam, cfg, trackers_map)
    return {'added': True, 'camera': cam}

@router.delete('/cameras/{cam_id}')
async def delete_camera(cam_id: int):
    global cams
    remaining = [c for c in cams if c['id'] != cam_id]
    if len(remaining) == len(cams):
        return {'error': 'Not found'}
    cams[:] = remaining
    stop_tracker(cam_id, trackers_map)
    save_cameras(cams, redis)
    return {'deleted': True}

@router.patch('/cameras/{cam_id}')
async def toggle_camera(cam_id: int):
    for cam in cams:
        if cam['id'] == cam_id:
            cam['enabled'] = not cam.get('enabled', True)
            if cam['enabled']:
                start_tracker(cam, cfg, trackers_map)
            else:
                stop_tracker(cam_id, trackers_map)
            save_cameras(cams, redis)
            return {'enabled': cam['enabled']}
    return {'error': 'Not found'}

@router.put('/cameras/{cam_id}')
async def update_camera(cam_id: int, request: Request):
    data = await request.json()
    for cam in cams:
        if cam['id'] == cam_id:
            if 'tasks' in data and isinstance(data['tasks'], list):
                tasks = data['tasks']
                mode = cam.get('mode', 'both')
                for m in ('in','out','both'):
                    if m in tasks:
                        mode = m
                        tasks = [t for t in tasks if t not in ('in','out','both')]
                        break
                cam['mode'] = mode
                cam['tasks'] = tasks
            if 'url' in data:
                cam['url'] = data['url']
            save_cameras(cams, redis)
            tr = trackers_map.get(cam_id)
            if tr:
                tr.update_cfg({'tasks': cam['tasks'], 'mode': cam['mode']})
            return {'updated': True}
    return {'error': 'Not found'}
