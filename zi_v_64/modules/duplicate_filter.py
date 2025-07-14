import cv2

class DuplicateFilter:
    """Simple duplicate frame detector using mean absolute difference."""
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.prev = None

    def is_duplicate(self, frame) -> bool:
        gray = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = gray
            return False
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        score = diff.mean() / 255 * 100
        return score < self.threshold
