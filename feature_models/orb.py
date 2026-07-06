import cv2
import numpy as np

def create():
    detector = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    return detector, matcher

def detect_and_compute_orb(img, detector):
    """Detect orb features"""
    feats = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), maxCorners=1000, qualityLevel=0.001, minDistance=25)
    feats = np.array([cv2.KeyPoint(x=int(f[0][0]), y=int(f[0][1]), size=10) for f in feats])
    kps, des = detector.compute(img, feats)
    return kps, des
