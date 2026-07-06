import streamlit as st
import numpy as np

from utils import *
st.title("Realsense T265 Into Route Conversion")
st.write("This page is for converting the bag files from Realsense T265 into latitude-longitude based route")

st.header("Bag file selection") 
annotation = st.file_uploader(
    "Choose annotation file",
    type=["bag"]
)
images_parent_path = st.text_input(
    "Test images path",
    r"D:/Code/Python/doctor code scratch/PyOrbSlam-main/bbox"
)

if (annotation and images_parent_path) is None:
    st.write("Please upload the json file first!")
else :
    st.write("Thanks for uploading it!")