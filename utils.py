import cv2
import numpy as np
import json
import re

from shapely.geometry import Polygon
from pathlib import Path
from feature_models.orb import detect_and_compute_orb
from feature_models.sift import detect_and_compute_sift
from feature_models.superpoint import detect_and_compute_superpoint, superpoint_match

def detect_and_compute(img, detector, kp_choice):

    if kp_choice == "ORB":
        return detect_and_compute_orb(img, detector)

    elif kp_choice == "SIFT":
        return detect_and_compute_sift(img, detector)

    elif kp_choice == "SuperPoint":
        return detect_and_compute_superpoint(img, detector)

    else:
        raise ValueError(f"Unknown keypoint: {kp_choice}")

def match_features(des1, des2, matcher):
    """Match features across consecutive frames"""
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def knn_match_features(des1, des2, matcher, ratio=0.75):
    """KNN matching with Lowe's ratio test"""

    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    return good_matches

def find_essential_matrix(matches, kp1, kp2, K):
    """Ransac filter for find essential matrix between kp matches"""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def recover_camera_pose(E, pts1, pts2, K, mask):
    """Estimate camera pose from essential matrix"""
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def transform_camera_pose(pts1, pts2):
    """Estimate camera pose from homography"""
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    return M

def extract_frame_number(filename):
    match = re.search(r"frame_(\d+)_jpg", filename)
    if match:
        return int(match.group(1))
    return float("inf")

def extract_frame_number(filename):
    match = re.search(r"frame_(\d+)_jpg", filename)
    if match:
        return int(match.group(1))
    return float("inf")


def load_coco_images_and_segmentations(file, parent_dir):
    """Open json file"""
    coco = json.load(file)

    seg_dict = {}

    # Take dict from annotation group
    for ann in coco["annotations"]:
        image_id = ann["image_id"]

        if image_id not in seg_dict:
            seg_dict[image_id] = []

        seg_dict[image_id].extend(ann["segmentation"])

    image_paths = []
    segmentations = []

    # Sort images by frame number
    sorted_images = sorted(
        coco["images"],
        key=lambda img: extract_frame_number(img["file_name"])
    )

    # Take dict from images group
    for img in sorted_images:
        image_id = img["id"]

        image_path = str(Path(parent_dir) / img["file_name"])

        image_paths.append(image_path)
        segmentations.append(seg_dict.get(image_id, []))

    # Convert segmentation coordinates into xy pair
    polygon_points = []

    for segs in segmentations:
        img_polygons = []

        for seg in segs:
            pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
            img_polygons.append(pts)

        polygon_points.append(img_polygons)

    return image_paths, polygon_points

def bbox_iou(boxA, boxB):
    """
    box = [xmin, ymin, xmax, ymax]
    """

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter

    if union == 0:
        return 0

    return inter / union

def polygon_iou(poly1_pts, poly2_pts):

    poly1 = Polygon(poly1_pts)
    poly2 = Polygon(poly2_pts)

    if not poly1.is_valid:
        poly1 = poly1.buffer(0)

    if not poly2.is_valid:
        poly2 = poly2.buffer(0)

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    if union == 0:
        return 0

    return intersection / union

def scale_polygons(polygon_points, scale):
    for i in range(len(polygon_points)):
        for j in range(len(polygon_points[i])):
            polygon_points[i][j] = [
                (x * scale, y * scale)
                for x, y in polygon_points[i][j]
            ]
    return polygon_points

def process_kp_slam(W,H,K,image_paths, polygon_points, kp_choice, detector, matcher):
    IOU_list = []
    pred_polygon_points = []
    result_images = []

    for idx, img_path in enumerate(image_paths):
        if idx == 1: continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (W, H))
        kp2, des2 = detect_and_compute(img, detector, kp_choice)
        gt_polygon = np.array(polygon_points[idx][0],dtype=np.float32)

        if idx == 0:
            pred_polygon = gt_polygon.copy()
            kp1 = kp2
            des1 = des2
            pred_polygon_points.append(pred_polygon)
            continue
        
        # Matching
        matches = match_features(des1, des2, matcher)
        # Estimate the essential matrix using matched features
        E, mask = find_essential_matrix(matches, kp1, kp2, K) # use RANSAC 
        # filter the matches based on the essential matrix mask
        filtered_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 2)
        # Recover camera pose using the filtered matches
        M = transform_camera_pose(pts1, pts2)
        pred_polygon = cv2.perspectiveTransform(pred_polygon.reshape(-1,1,2),M).reshape(-1,2)

        # Clamp coordinate
        pred_polygon[:,0] = np.clip(pred_polygon[:,0],0,W - 1)
        pred_polygon[:,1] = np.clip(pred_polygon[:,1],0,H - 1)
        pred_polygon_points.append(pred_polygon)

        # IoU
        iou = polygon_iou(pred_polygon,gt_polygon)
        IOU_list.append(iou)

        # Save image
        vis = img.copy()
        cv2.polylines(vis,[gt_polygon.astype(np.int32)],True,(0,255,0),5)
        cv2.polylines(vis,[pred_polygon.astype(np.int32)],True,(0,0,255),5)
        cv2.putText(vis,f"IOU={iou:.3f}",(50,80),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        result_images.append(vis_rgb)

        # Update
        kp1 = kp2
        des1 = des2

    IOU_array = np.array(IOU_list)
    return {
        "images": result_images,
        "pred_polygon_points": pred_polygon_points,
        "iou": IOU_array,
        "mean_iou": float(IOU_array.mean()),
        "median_iou": float(np.median(IOU_array)),
        "min_iou": float(IOU_array.min()),
        "max_iou": float(IOU_array.max()),
    }

def process_superpoint_slam(
    W,
    H,
    K,
    image_paths,
    polygon_points,
    matching,
):

    IOU_list = []
    pred_polygon_points = []
    result_images = []

    pred_polygon = None

    for idx, img_path in enumerate(image_paths):

        if idx == 1:
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (W, H))

        gt_polygon = np.array(
            polygon_points[idx][0],
            dtype=np.float32
        )

        if idx == 0:

            pred_polygon = gt_polygon.copy()

            img_prev = img.copy()

            pred_polygon_points.append(pred_polygon)

            continue

        ###################################################
        # SuperPoint + SuperGlue
        ###################################################

        pts1, pts2 = superpoint_match(
            img_prev,
            img,
            matching
        )

        ###################################################

        M = transform_camera_pose(
            pts1,
            pts2
        )

        pred_polygon = cv2.perspectiveTransform(
            pred_polygon.reshape(-1,1,2),
            M
        ).reshape(-1,2)

        pred_polygon[:,0] = np.clip(pred_polygon[:,0],0,W-1)
        pred_polygon[:,1] = np.clip(pred_polygon[:,1],0,H-1)

        pred_polygon_points.append(pred_polygon)

        iou = polygon_iou(
            pred_polygon,
            gt_polygon
        )

        IOU_list.append(iou)

        vis = img.copy()

        cv2.polylines(
            vis,
            [gt_polygon.astype(np.int32)],
            True,
            (0,255,0),
            5
        )

        cv2.polylines(
            vis,
            [pred_polygon.astype(np.int32)],
            True,
            (0,0,255),
            5
        )

        cv2.putText(
            vis,
            f"IOU={iou:.3f}",
            (50,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255,0,0),
            3
        )

        result_images.append(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        )

        img_prev = img.copy()

    IOU_array = np.array(IOU_list)

    return {
        "images": result_images,
        "pred_polygon_points": pred_polygon_points,
        "iou": IOU_array,
        "mean_iou": float(IOU_array.mean()),
        "median_iou": float(np.median(IOU_array)),
        "min_iou": float(IOU_array.min()),
        "max_iou": float(IOU_array.max()),
    }