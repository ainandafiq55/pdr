import datetime
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Show app title and description.
st.set_page_config(page_title="Pedestrian Dead Reckoning", page_icon="👨‍🦯‍➡️")

pages = [
    st.Page("home.py", title="Home", icon="🏡"),
    st.Page("kp_analytics.py", title="Keypoint SLAM Analytics", icon="📈"),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
