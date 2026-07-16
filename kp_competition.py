import streamlit as st
import numpy as np
import cv2 
import inspect

from utils import *
st.title("Keypoints Competition")
st.write("This page is to select which keypoint is perform best")

st.header("Reference image selection") 

start_image_name = st.text_input(
    "Type the name of the first starting frame that has been annotated",
    r"frame_0_jpg.rf.d61fa0abf446845c21e89849ebf46358.jpg"
)
end_image_name = st.text_input(
    "Type the name of the last ending frame that has been annotated",
    r"frame_369_jpg.rf.28d5a1955d52f033f8b0599870152f73.jpg"
)

reference_image_annotation = st.file_uploader(
    "Choose the bbox annotation for the reference images",
    type=["json"]
)

st.header("Frames selection") 
st.write("Please put image frames folder during walking after the starting image until the end of walking")
image_frames_folder = st.text_input(
    "Image frames path",
    r"D:/Media/doctor related/pdr record frames/lab-toilet_back-camera-face-front/frame_2026-04-13-14-42-44-371"
)
end_frame_name = st.text_input(
    "Type the name of the last ending frame (most likely same with the annotated ending frame)",
    r"frame_369.jpg"
)

if all(x is not None for x in (start_image_name, end_image_name, reference_image_annotation, image_frames_folder)):
    st.write("Please upload the json file first!")
else :
    # Get test polygon points for start and end images
    start_image_polygon_points = get_polygon_points(start_image_name,reference_image_annotation)
    end_image_polygon_points = get_polygon_points(end_image_name,reference_image_annotation)

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
        if kp_choice == "SuperPoint":
            # Superpoint based proces
            from feature_models.superpoint import create
            matching = create()
            
            polygon_points = scale_polygons(polygon_points, scale)
            result = process_superpoint_slam(W=W,H=H,K=K,image_paths=image_paths,polygon_points=polygon_points,matching=matching,)

        else :
            if kp_choice == "ORB" :
                from feature_models.orb import create

            elif kp_choice == "SIFT":
                from feature_models.sift import create
            
            detector, matcher = create()

            # Drawn images results, Mean IOU, median, min, max
            polygon_points = scale_polygons(polygon_points, scale)
            result = process_kp_slam(W,H,K,image_paths, polygon_points, kp_choice, detector, matcher)

        st.metric("Mean IoU", f"{result['mean_iou']:.3f}")
        st.metric("Median IoU", f"{result['median_iou']:.3f}")
        st.metric("Min IoU", f"{result['min_iou']:.3f}")
        st.metric("Max IoU", f"{result['max_iou']:.3f}")

        for img in result["images"]:
            st.image(img)