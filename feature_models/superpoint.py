import cv2
import torch
import torch
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
# SuperGluePretrainedNetwork\models\matching.py

def create(device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "superpoint": {
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 1024,
        },
        "superglue": {
            "weights": "indoor",
            "sinkhorn_iterations": 20,
            "match_threshold": 0.2,
        },
    }

    matching = Matching(config).eval().to(device)

    return matching

def image2tensor(img, device):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tensor = torch.from_numpy(gray/255.).float()[None,None]

    return tensor.to(device)

def detect_and_compute_superpoint(img, matching):

    device = next(matching.parameters()).device

    image_tensor = image2tensor(img, device)

    pred = matching.superpoint(
        {"image": image_tensor}
    )

    kp = pred["keypoints"][0]
    des = pred["descriptors"][0]

    return kp, des

def superpoint_match(img1, img2, matching):

    device = next(matching.parameters()).device

    inp0 = image2tensor(img1, device)
    inp1 = image2tensor(img2, device)

    with torch.no_grad():
        pred = matching({
            "image0": inp0,
            "image1": inp1,
        })

    kpts0 = pred["keypoints0"][0].cpu().numpy()
    kpts1 = pred["keypoints1"][0].cpu().numpy()

    matches = pred["matches0"][0].cpu().numpy()

    valid = matches > -1

    pts1 = kpts0[valid]
    pts2 = kpts1[matches[valid]]

    # Freeing tensors
    del pred
    del inp0
    del inp1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pts1.astype(np.float32), pts2.astype(np.float32)