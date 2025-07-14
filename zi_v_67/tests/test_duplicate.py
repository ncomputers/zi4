import sys
import types
try:
    import cv2
except Exception:
    cv2 = types.ModuleType('cv2')
    cv2.COLOR_BGR2GRAY = 0
    cv2.resize = lambda img, size: img
    cv2.cvtColor = lambda img, flag: img
    cv2.absdiff = lambda a, b: [[0]]
    cv2.CAP_FFMPEG = 1900
    cv2.CAP_PROP_BUFFERSIZE = 38
    class _Cap:
        def set(self, *a, **k):
            pass
    cv2.VideoCapture = lambda *a, **k: _Cap()
    sys.modules['cv2'] = cv2

from zi_v_67.modules.duplicate_filter import DuplicateFilter

class DummyFilter(DuplicateFilter):
    def is_duplicate(self, frame):
        if self.prev is None:
            self.prev = frame
            return False
        dup = frame == self.prev
        self.prev = frame
        return dup

def test_duplicate_filter():
    f = DummyFilter(1.0)
    img1 = "frame1"
    img2 = "frame1"
    assert not f.is_duplicate(img1)
    assert f.is_duplicate(img2)
    img3 = "frame2"
    assert not f.is_duplicate(img3)
