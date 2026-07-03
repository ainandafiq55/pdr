import streamlit as st
import numpy as np
import cv2 
import inspect

from utils import *
st.title("Keypoint SLAM Analytics")
st.write("This page is for analyze the keypoints performance")

st.header("Annotation selection") 
annotation = st.file_uploader(
    "Choose annotation file",
    type=["json"]
)
images_parent_path = st.text_input(
    "Test images path",
    r"D:/Code/Python/doctor code scratch/PyOrbSlam-main/bbox"
)

if (annotation and images_parent_path) is None:
    st.write("Please upload the json file first!")
else :
    # Get test images path
    image_paths, polygon_points = load_coco_images_and_segmentations(annotation,images_parent_path)

    # Camera instrinsic features
    st.header("Camera Instrinsic Value Settings") 
    st.write("K value")
    col1, col2 = st.columns(2)

    with col1:
        W_input = st.number_input(
            "Width",
            min_value=1,
            value=2160,
            step=1
        )

    with col2:
        H_input = st.number_input(
            "Height",
            min_value=1,
            value=3840,
            step=1
        )
    st.write("K value")
    default_K = [
        [2671.63477, 0.0, 1031.23813],
        [0.0, 2684.31749, 1873.57076],
        [0.0, 0.0, 1.0]
    ]

    K_input = np.zeros((3, 3))
    for i in range(3):
        cols = st.columns(3)

        for j in range(3):
            with cols[j]:
                K_input[i, j] = st.number_input(
                    f"K[{i},{j}]",
                    value=float(default_K[i][j]),
                    key=f"k_{i}_{j}",
                    label_visibility="collapsed"
                )
    scale = st.number_input(
            "Camera intrinsic reducing ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    K = K_input.copy()
    K[:2, :] *= scale
    K[2, 2] = 1.0
    W = int(W_input * scale)
    H = int(H_input * scale)
    
    # Keypoint selection part
    st.header("Keypoint selection") 
    kp_choice = st.selectbox(
        "Choose the keypoint extractor:", 
        ["ORB", "SIFT", "SuperPoint"])

    matcher_options = {
        "ORB": ["NORM_HAMMING"],
        "SIFT": ["FlannBased"],
        "SuperPoint": ["SuperGlue"]
    }

    matcher_input = st.selectbox(
        "Matcher",
        matcher_options[kp_choice]
    )

    detector = None
    if st.button("Process"):
        if kp_choice == "ORB" :
            detector = cv2.ORB_create()
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        elif kp_choice == "SIFT":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            detector = cv2.SIFT_create()
            matcher = cv2.FlannBasedMatcher(index_params, search_params)

        elif kp_choice == "SuperPoint":
            ## TODO: add superpoint
            detector = None
            matcher = None
        
        if detector is None:
            st.write("Coming in future update")

        else:
            # Drawn images results, Mean IOU, median, min, max
            polygon_points = scale_polygons(polygon_points, scale)
            result = process_kp_slam(W,H,K,image_paths, polygon_points, kp_choice, detector, matcher)

            st.metric("Mean IoU", f"{result['mean_iou']:.3f}")
            st.metric("Median IoU", f"{result['median_iou']:.3f}")
            st.metric("Min IoU", f"{result['min_iou']:.3f}")
            st.metric("Max IoU", f"{result['max_iou']:.3f}")

            for img in result["images"]:
                st.image(img)