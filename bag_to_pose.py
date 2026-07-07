import streamlit as st
import pandas as pd

from utils import process_bag2pose
from streamlit_folium import st_folium
from pathlib import Path

st.title("Realsense T265 Into Route Conversion")
st.write("This page is for converting the bag files from Realsense T265 into latitude-longitude based route")

if "map" not in st.session_state:
    st.session_state.map = None

if "route" not in st.session_state:
    st.session_state.route = None

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
        value=33.889350,
        step=0.000001
    )

with col2:
    start_lon = st.number_input(
        "Starting Longitude",
        min_value=-180.00,
        value=130.710150,
        step=0.000001
    )
start_azimuth = st.number_input(
        "Starting Azimuth",
        min_value=0.00,
        value=2.20,
        step=0.001
    )

if st.button("Process"):
    if not Path(bag_path).exists():
        st.error("Bag file not found.")
    else:
        with st.spinner("Processing..."):
            m, route = process_bag2pose(bag_path, start_lat, start_lon, start_azimuth)
            st.session_state.map = m
            st.session_state.route = route

if st.session_state.map is not None:

    st.success("Route has been predicted!")
    st_folium(
        st.session_state.map,
        width=900,
        height=600,
        key="route_map",
        returned_objects=[]
    )