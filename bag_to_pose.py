import streamlit as st

from utils import process_bag2pose, load_groundtruth_geojson, calculate_ate, create_map, calculate_polyline_error
from streamlit_folium import st_folium
from pathlib import Path

st.title("Realsense T265 Into Route Conversion")
st.write("This page is for converting the bag files from Realsense T265 into latitude-longitude based route")

if "map" not in st.session_state:
    st.session_state.map = None

if "bag_map" not in st.session_state:
    st.session_state.bag_map = None

if "route" not in st.session_state:
    st.session_state.route = None

if "gt_route" not in st.session_state:
    st.session_state.gt_route = None

if "ate_error" not in st.session_state:
    st.session_state.ate_error = None

if "poly_error" not in st.session_state:
    st.session_state.poly_error = None

st.header("Bag file selection") 
bag_path = st.text_input(
    "Bag file path",
    r"D:/Media/doctor related/try realsense tracking/lab-toilet.bag"
)
col1, col2 = st.columns(2)
with col1:
    start_lat = st.number_input(
        "Starting Latitude",
        min_value=-90.00,
        max_value=90.0,
        value=33.889392,
        step=0.000001,
        format="%.6f"
    )

with col2:
    start_lon = st.number_input(
        "Starting Longitude",
        min_value=-180.00,
        max_value=180.0,
        value=130.710155,
        step=0.000001,
        format="%.6f"
    )
start_azimuth = st.number_input(
        "Starting Azimuth",
        min_value=0.00,
        max_value=90.0,
        value=2.20,
        step=0.001,
        format="%.3f"
    )

st.header("Map annotation input") 
geojson = st.file_uploader(
    "Choose .geojson file",
    type=["geojson"]
)

if geojson is not None:
    gt_route = load_groundtruth_geojson(geojson)
    st.session_state.gt_route = gt_route
else :
    st.session_state.gt_route = None

if st.button("Process"):
    if not Path(bag_path).exists():
        st.error("Bag file not found.")
    else:
        with st.spinner("Processing..."):
            bag_m, route = process_bag2pose(bag_path, start_lat, start_lon, start_azimuth)
            st.session_state.bag_map = bag_m
            st.session_state.map = bag_m
            st.session_state.route = route

if st.session_state.route is not None:
    st.success("Route has been predicted!")
    m = create_map(st.session_state.route,st.session_state.gt_route)
    st_folium(
        m,
        width=900,
        height=600,
        key="route_map",
        returned_objects=[]
    )
    
    if st.session_state.gt_route is not None:
        ate_err = calculate_ate(st.session_state.route,st.session_state.gt_route)
        poly_err = calculate_polyline_error(st.session_state.route,st.session_state.gt_route)
        st.session_state.ate_error = ate_err
        st.session_state.poly_error = poly_err
        
        if (st.session_state.ate_error and st.session_state.poly_error) is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Polyline Error")
                st.metric("Mean", f"{st.session_state.poly_error['mean']:.2f} m")
                st.metric("RMSE", f"{st.session_state.poly_error['rmse']:.2f} m")
                st.metric("Max", f"{st.session_state.poly_error['max']:.2f} m")

            with col2:
                st.write("### ATE")
                st.metric("Mean", f"{st.session_state.ate_error['mean']:.2f} m")
                st.metric("RMSE", f"{st.session_state.ate_error['rmse']:.2f} m")
                st.metric("Max", f"{st.session_state.ate_error['max']:.2f} m")
                
            