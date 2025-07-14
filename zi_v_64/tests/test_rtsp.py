import sys
import types
try:
    import cv2
except Exception:  # pragma: no cover - skip if OpenCV not installed
    import types
    cv2 = types.ModuleType('cv2')
    cv2.CAP_FFMPEG = 1900
    cv2.CAP_PROP_BUFFERSIZE = 38
    def dummy_vc(*a, **k):
        class Cap:
            def set(self, *a, **k):
                pass
        return Cap()
    cv2.VideoCapture = dummy_vc
    sys.modules['cv2'] = cv2

from unittest import mock

if 'torch' not in sys.modules:
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)))
    sys.modules['torch'] = torch

if 'ultralytics' not in sys.modules:
    ul = types.ModuleType('ultralytics')
    ul.YOLO = lambda *a, **k: DummyYOLO()
    sys.modules['ultralytics'] = ul

if 'loguru' not in sys.modules:
    logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    sys.modules['loguru'] = types.ModuleType('loguru')
    sys.modules['loguru'].logger = logger

if 'deep_sort_realtime.deepsort_tracker' not in sys.modules:
    ds_mod = types.ModuleType('deep_sort_realtime.deepsort_tracker')
    ds_mod.DeepSort = mock.MagicMock()
    sys.modules['deep_sort_realtime'] = types.ModuleType('deep_sort_realtime')
    sys.modules['deep_sort_realtime.deepsort_tracker'] = ds_mod

if 'redis' not in sys.modules:
    redis_stub = types.ModuleType('redis')
    redis_stub.Redis = mock.MagicMock(from_url=mock.MagicMock(return_value=mock.MagicMock(get=lambda *a, **k: None, mset=lambda *a, **k: None, incr=lambda *a, **k: None, rpush=lambda *a, **k: None)))
    sys.modules['redis'] = redis_stub

if 'core.config' not in sys.modules:
    cfg_mod = types.ModuleType('core.config')
    cfg_mod.ANOMALY_ITEMS = []
    sys.modules['core'] = types.ModuleType('core')
    sys.modules['core.config'] = cfg_mod

from zi_v_64.modules.tracker import FlowTracker

class DummyYOLO:
    names = ['person']
    def __init__(self, *a, **k):
        self.model = mock.MagicMock()
    def predict(self, *a, **k):
        return [mock.MagicMock(boxes=mock.MagicMock(data=[]))]

if 'torch' not in sys.modules:
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False), backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)))
    sys.modules['torch'] = torch

if 'ultralytics' not in sys.modules:
    ul = types.ModuleType('ultralytics')
    ul.YOLO = DummyYOLO
    sys.modules['ultralytics'] = ul

def make_tracker(src_type):
    with mock.patch('zi_v_64.modules.tracker.YOLO', DummyYOLO), \
         mock.patch('zi_v_64.modules.tracker.redis.Redis.from_url') as rmock:
        rmock.return_value = mock.MagicMock(get=lambda *a: None, mset=lambda *a, **kw: None)
        cfg = {'redis_url':'redis://localhost:6379/0','person_model':'m.pt','ppe_model':'m.pt'}
        tr = FlowTracker(1, 'stream', ['person'], cfg, [])
        tr.src_type = src_type
        return tr

def test_open_capture_rtsp():
    if cv2 is None:
        return
    tr = make_tracker('rtsp')
    with mock.patch('cv2.VideoCapture') as cap:
        tr._open_capture()
        cap.assert_called_with('stream', cv2.CAP_FFMPEG)

def test_open_capture_http():
    if cv2 is None:
        return
    tr = make_tracker('http')
    with mock.patch('cv2.VideoCapture') as cap:
        tr._open_capture()
        cap.assert_called_with('stream')
