import time
import folium
import math
import cv2
import numpy as np
import json
import re
import pyrealsense2 as rs

from shapely.geometry import Polygon, LineString,Point
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

def predict_polygon_pose(origin_polygon, W, H, M):
    """Convert the polygon pose based on camera pose matrix and clamp it"""
    pred_polygon = cv2.perspectiveTransform(origin_polygon.reshape(-1,1,2),M).reshape(-1,2)

    # Clamp coordinate
    pred_polygon[:,0] = np.clip(pred_polygon[:,0],0,W - 1)
    pred_polygon[:,1] = np.clip(pred_polygon[:,1],0,H - 1)

    return pred_polygon
def match_and_transform_pose(kp1, kp2, des1, des2, matcher, K):
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
        num_match = len(filtered_matches)
        return M, num_match 

def process_kp_slam(W,H,K,image_paths, polygon_points, kp_choice, detector, matcher):
    IOU_list = []
    pred_polygon_points = []
    result_images = []
    process_times = []
    blur_list = []
    num_match_list = []

    for idx, img_path in enumerate(image_paths):
        # if idx == 1: continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (W, H))
        start_time = time.perf_counter()
        kp2, des2 = detect_and_compute(img, detector, kp_choice)
        gt_polygon = np.array(polygon_points[idx][0],dtype=np.float32)

        if idx == 0:
            pred_polygon = gt_polygon.copy()
            kp1 = kp2
            des1 = des2
            pred_polygon_points.append(pred_polygon)
            continue
        
        M, num_match = match_and_transform_pose(kp1, kp2, des1, des2, matcher, K)
        end_time = time.perf_counter()

        num_match_list.append(num_match)

        pred_polygon = predict_polygon_pose(pred_polygon, W, H, M)
        pred_polygon_points.append(pred_polygon)

        # IoU
        iou = polygon_iou(pred_polygon,gt_polygon)
        IOU_list.append(iou)

        # Time
        elapsed_time = end_time - start_time
        process_times.append(elapsed_time)

        # Laplacian
        blur = variance_of_laplacian(img)
        blur_list.append(blur)

        vis = img.copy()
        cv2.polylines(vis,[gt_polygon.astype(np.int32)],True,(0,255,0),5)
        cv2.polylines(vis,[pred_polygon.astype(np.int32)],True,(0,0,255),5)
        cv2.putText(vis,f"IOU={iou:.3f}",(50,80),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Blur={blur:.3f}",(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Matches KP={num_match}",(50,220),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Time ={elapsed_time:.3f}",(50,290),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        result_images.append(vis_rgb)

        # Update
        kp1 = kp2
        des1 = des2

    IOU_array = np.array(IOU_list)
    process_times = np.array(process_times)
    blur_array = np.array(blur_list)
    return {
        "images": result_images,
        "pred_polygon_points": pred_polygon_points,
        "iou": IOU_array,
        "mean_iou": float(IOU_array.mean()),
        "median_iou": float(np.median(IOU_array)),
        "min_iou": float(IOU_array.min()),
        "max_iou": float(IOU_array.max()),
        "lowest_blur":float(blur_array.min()),
        "average_time": float(process_times.mean()),
        "median_time":float(np.median(process_times)),
        "blur_list":blur_list,
        "match_list":num_match_list,
    }

def process_superpoint_slam(W,H,K,image_paths,polygon_points,matching,):

    IOU_list = []
    pred_polygon_points = []
    result_images = []
    process_times = []
    blur_list = []
    num_match_list = []

    pred_polygon = None

    for idx, img_path in enumerate(image_paths):

        # if idx == 1:
        #     continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (W, H))

        gt_polygon = np.array(polygon_points[idx][0],dtype=np.float32)

        if idx == 0:
            pred_polygon = gt_polygon.copy()
            img_prev = img.copy()
            pred_polygon_points.append(pred_polygon)
            continue

        # SuperPoint + SuperGlue
        start_time = time.perf_counter()
        pts1, pts2 = superpoint_match(img_prev,img,matching)
        num_match = len(pts1)
        M = transform_camera_pose(pts1,pts2)
        end_time = time.perf_counter()
        pred_polygon = predict_polygon_pose(pred_polygon, W, H, M)
        pred_polygon_points.append(pred_polygon)
        
        # Number of matches
        num_match_list.append(num_match)

        # IOU
        iou = polygon_iou(pred_polygon,gt_polygon)
        IOU_list.append(iou)
        vis = img.copy()

        # Time
        elapsed_time = end_time - start_time
        process_times.append(elapsed_time)

        # Laplacian
        blur = variance_of_laplacian(img)
        blur_list.append(blur)

        cv2.polylines(vis,[gt_polygon.astype(np.int32)],True,(0,255,0),5)
        cv2.polylines(vis,[pred_polygon.astype(np.int32)],True,(0,0,255),5)
        cv2.putText(vis,f"IOU={iou:.3f}",(50,80),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Blur={variance_of_laplacian(img):.3f}",(50,150),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Matches KP={num_match}",(50,220),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
        cv2.putText(vis,f"Time ={elapsed_time:.3f}",(50,290),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)

        result_images.append(
            cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        )

        img_prev = img.copy()

    IOU_array = np.array(IOU_list)
    process_times = np.array(process_times)
    blur_array = np.array(blur_list)

    return {
        "images": result_images,
        "pred_polygon_points": pred_polygon_points,
        "iou": IOU_array,
        "mean_iou": float(IOU_array.mean()),
        "median_iou": float(np.median(IOU_array)),
        "min_iou": float(IOU_array.min()),
        "max_iou": float(IOU_array.max()),
        "lowest_blur":float(blur_array.min()),
        "average_time": float(process_times.mean()),
        "median_time":float(np.median(process_times)),
        "blur_list":blur_list,
        "match_list":num_match_list,
    }

def process_bag2pose(bag_path, start_lat, start_lon, start_azimuth):
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    poses = []

    try:        
        print("Reading bag...")
        while True:
            frames = pipeline.wait_for_frames()

            pose = frames.get_pose_frame()

            if pose:
                data = pose.get_pose_data()

                poses.append([
                    data.translation.x,
                    data.translation.y,
                    data.translation.z,
                    data.rotation.w,
                    data.rotation.x,
                    data.rotation.y,
                    data.rotation.z
                ])

    except RuntimeError:
        pass

    pipeline.stop()
    print("Finished reading")

    ## Convert to route
    lat = start_lat
    lon = start_lon

    lat_per_meter = 1 / 111320
    lon_per_meter = 1 / (111320 * math.cos(math.radians(start_lat)))

    theta = start_azimuth

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    lat_per_meter = 1 / 111320
    lon_per_meter = 1 / (111320 * math.cos(math.radians(start_lat)))

    route = []
    for p in poses:

        x = p[0]
        z = p[2]

        # Rotate local coordinate
        forward = -z
        right = x

        east = forward * sin_t + right * cos_t
        north = forward * cos_t - right * sin_t

        lat = start_lat + north * lat_per_meter
        lon = start_lon + east * lon_per_meter

        route.append([lat, lon])
    # Draw folium
    m = folium.Map(location=route[0],zoom_start=18, tiles='https://cyberjapandata.gsi.go.jp/xyz/ort/{z}/{x}/{y}.jpg', attr='GSI')
    folium.PolyLine(route,color="blue",weight=4).add_to(m)
    folium.Marker(route[0],tooltip="Start",icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(route[-1],tooltip="Finish",icon=folium.Icon(color="red")).add_to(m)
    print("Sending back to the UI")
    return m, route

def load_groundtruth_geojson(geojson_file):
    """
    Read LineString from GeoJSON.
    Returns:
        gt_route = [[lat, lon], ...]
    """

    if geojson_file is None:
        return None

    data = json.load(geojson_file)

    gt_route = []

    for feature in data["features"]:

        if feature["geometry"]["type"] != "LineString":
            continue

        for lon, lat in feature["geometry"]["coordinates"]:
            gt_route.append([lat, lon])

    return gt_route

def draw_groundtruth(m, gt_route):

    if gt_route is None:
        return m

    folium.PolyLine(
        gt_route,
        color="red",
        weight=4,
        tooltip="Ground Truth"
    ).add_to(m)

    return m

def calculate_route_error(pred_route, gt_route):
    """
    Mean distance (meter) from each predicted point
    to the nearest ground truth point.
    """

    if pred_route is None or gt_route is None:
        return None

    errors = []

    for p in pred_route:

        plat, plon = p

        dmin = float("inf")

        for g in gt_route:

            glat, glon = g

            dlat = (plat - glat) * 111320
            dlon = (plon - glon) * 111320 * np.cos(np.radians(glat))

            d = np.sqrt(dlat**2 + dlon**2)

            dmin = min(dmin, d)

        errors.append(dmin)
    print(errors)
    return {
        "mean": np.mean(errors),
        "max": np.max(errors),
        "rmse": np.sqrt(np.mean(np.square(errors)))
    }

def create_map(route, gt_route=None):
    m = folium.Map(
        location=route[0],
        zoom_start=18,
        tiles='https://cyberjapandata.gsi.go.jp/xyz/ort/{z}/{x}/{y}.jpg',
        attr='GSI'
    )

    folium.PolyLine(
        route,
        color="blue",
        weight=4
    ).add_to(m)

    folium.Marker(route[0],tooltip="Start",icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(route[-1],tooltip="Finish",icon=folium.Icon(color="red")).add_to(m)

    if gt_route is not None:

        folium.PolyLine(
            gt_route,
            color="red",
            weight=4
        ).add_to(m)

    return m

def latlon_to_xy(route):
    """
    Convert latitude-longitude to local XY coordinates (meters).
    X = East
    Y = North
    """

    lat0, lon0 = route[0]

    xy = []

    for lat, lon in route:

        x = (lon - lon0) * 111320 * np.cos(np.radians(lat0))
        y = (lat - lat0) * 111320

        xy.append([x, y])

    return np.asarray(xy)

def calculate_polyline_error(pred_route, gt_route):
    """
    Point-to-polyline distance.
    Returns error in meters.
    """

    pred_xy = latlon_to_xy(pred_route)
    gt_xy = latlon_to_xy(gt_route)

    gt_line = LineString(gt_xy)

    errors = []

    for p in pred_xy:

        pt = Point(p)

        errors.append(pt.distance(gt_line))

    errors = np.asarray(errors)

    return {
        "mean": errors.mean(),
        "rmse": np.sqrt(np.mean(errors**2)),
        "max": errors.max()
    }

def resample_route(route, n_points):
    """
    Uniform resampling.
    """

    route = np.asarray(route)

    old_idx = np.linspace(0, 1, len(route))
    new_idx = np.linspace(0, 1, n_points)

    lat = np.interp(new_idx, old_idx, route[:,0])
    lon = np.interp(new_idx, old_idx, route[:,1])

    return np.column_stack((lat, lon))

def calculate_ate(pred_route, gt_route):

    n = max(len(pred_route), len(gt_route))

    pred = resample_route(pred_route, n)
    gt = resample_route(gt_route, n)

    pred_xy = latlon_to_xy(pred)
    gt_xy = latlon_to_xy(gt)

    errors = np.linalg.norm(pred_xy - gt_xy, axis=1)

    return {
        "mean": errors.mean(),
        "rmse": np.sqrt(np.mean(errors**2)),
        "max": errors.max()
    }

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    gray = img2gray(image)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_polygon_points(image_name, annotation):
    # Find based on image_id
    image_id = next(
        (img["id"] for img in annotation["images"]
         if img["extra"]["name"] == image_name),
        None
    )
    if image_id is None:
        return []
    
    # Look for annotation based on image_id
    ann = next(
        (ann for ann in annotation["annotations"]
         if ann["image_id"] == image_id),
        None
    )
    if ann is None:
        return []
    
    polygon_points = []
    for seg in ann["segmentation"]:
        pts = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
        polygon_points.append(pts)

    return polygon_points

def img2gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def get_new_pose_kp(img, W, H, K, polygon_points, kp_choice, detector, matcher, kp1, des1):
    img = cv2.resize(img, (W, H))
    kp2, des2 = detect_and_compute(img, detector, kp_choice)
    
    M = match_and_transform_pose(kp1, kp2, des1, des2, matcher, K)
    img = cv2.resize(img, (W, H))
    kp2, des2 = detect_and_compute(img, detector, kp_choice)
    return polygon_points