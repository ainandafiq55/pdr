
import cv2

def create():
    detector = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    return detector, matcher

def detect_and_compute_sift(img, detector):
    """Detect sift features"""
    kps, des = detector.detectAndCompute(img, None)
    return kps, des