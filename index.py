import flood_index
import wri
import home
import dailyforecast
import hourlyforecast
import streamlit as st 

PAGES = {
    "Getting Started" : home,
    "Daily Flood Index": flood_index,
    "Hourly Water Resources Index": wri,
    "Daily Flood Forecast (Demo)": dailyforecast,
    "Hourly Flood Forecast (Demo)": hourlyforecast
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("", list(PAGES.keys()))

page = PAGES[selection]
page.app()