import cv2
import time

class DuplicateFilter:
    """Simple duplicate frame detector using mean absolute difference with bypass."""
    def __init__(self, threshold: float = 0.1, bypass_seconds: int = 2):
        self.threshold = threshold
        self.bypass_seconds = bypass_seconds
        self.prev = None
        self.bypass_until = 0.0

    def is_duplicate(self, frame) -> bool:
        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = gray
            return False
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        score = diff.mean() / 255 * 100
        if score >= self.threshold:
            self.bypass_until = time.time() + self.bypass_seconds
            return False
        if time.time() < self.bypass_until:
            return False
        return True
