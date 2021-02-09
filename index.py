import flood_index
import wri
import streamlit as st 

PAGES = {
    "Daily Flood Index": flood_index,
    "Hourly Water Resources Index": wri
}

st.sidebar.title('Available Tools')
selection = st.sidebar.radio("Compute", list(PAGES.keys()))
st.sidebar.title('Computation Details')
page = PAGES[selection]
page.app()