import streamlit as st

# Show app title and description.
st.set_page_config(page_title="Pedestrian Dead Reckoning", page_icon="👨‍🦯‍➡️")

pages = [
    st.Page("home.py", title="Home", icon="🏡"),
    st.Page("kp_analytics.py", title="Keypoint SLAM Analytics", icon="📈"),
    st.Page("bag_to_pose.py", title="Realsense T265 Into Route Conversion", icon="🗺️"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
