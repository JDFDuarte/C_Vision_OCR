# _______ Library Import _______ #
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from datetime import datetime
import base64
from PIL import Image
from PIL import Image
import markdown_functions as md


# _______ Page Setup _______ #
st.set_page_config(
    page_title="NASA Weather Exploration",
    page_icon="🌝",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# _______ Functions _______ #
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


background_image = "images/background.jpg"
encoded_image = get_base64_of_bin_file(background_image)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown(md.get_navbar_markdown(), unsafe_allow_html=True)
st.markdown(md.background_and_tabs_styles(encoded_image), unsafe_allow_html = True)



# _______ Page _______ #
st.write("")
st.title("")
col_1, col_2, col_3 = st.columns([8, 2, 0.5])

with col_1:
    st.title("")
    st.title("About")

with col_2:
    st.title("")
    st.image("images/logo_cs.png", width=300)


st.write("")
st.write("")


tab1, tab2, tab3, tab4 = st.tabs(["**About Us**", "**Our Team**", "**Data**", "**Analysis**"])


# adjusts tab placement (navbar flushed it to the right, this pushes it to the left)
st.markdown(
    """
    <style>
    .stTabs [role="tablist"] {
        justify-content: flex-start;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# text in the About page is separated in 4 tabs 
with tab1:
    st.title("")
    st.subheader("Who are we?")
    st.write("")
    st.write("""Our small group is composed of three space aficionados with Data as their first love 🧡").
             <br>
             We met because these two interests brought us together. And we very quickly agreed that good and relevant data about Mars and even Earth is not only hard to find,
             it is not easily accessible to everyone in the community.
             As action takers we decided to take matters into our own hands and came up with Comparing Skies to break these barriers and 
             drive space exploration. """, unsafe_allow_html=True)

with tab2:
    st.write("")


with tab3:
    st.title("")
    st.write("Our data is comprised of:")
    
    col_1, col_2, col_3, col_4, col_5 = st.columns([3.5, 0.5, 3.5, 0.5, 3.5])

    with col_1:
        st.markdown("**Earth's Weather Data**")
        container = st.container(border=True)
        container.write("""This data was obtained from Open-Meteo's API. This allows us to obtain data from 1940 until the present day.
                        The dataset contains information about:
                        \n 🌡️ Air temperature at 2 meters above ground (ºC) 
                        \n 🌧️ Only liquid precipitation of the preceding hour including local showers and rain from large scale systems (mm)
                        \n 🌤️ Direct solar radiation 
                        \n 💦 Relative humidity at 2 meters above ground (%)""")

    with col_2:
        st.write("")

    with col_3:
        st.markdown("**Mars' Weather Data**")
        container = st.container(border=True)
        container.write("""This data was obtained from [Kaggle](https://www.kaggle.com/datasets/thedevastator/mars-weather-data-from-2012-to-2018). It contains data from 2012 to 2018 and we're working hard to collect more data to be added in the future.
                        The dataset holds information about:
                        \n 📆 Terrestrial date ↔️ The date on Earth when the data was collected 
                        \n 🌧️ Martian day when the data was collected
                        \n ⚖️ Min and max temperature
                        \n ↘️ Atmospheric pressure
                        \n 🌬️ Wind speed
                        """)

    with col_4:
        st.write("")

    with col_5:
        st.markdown("**Solar Flares Data**")
        container = st.container(border=True)
        container.write("""This data was obtained from NASA's DONKI (The Space Weather Database Of Notifications, Knowledge, Information). .
                        The dataset contains information about:
                        \n ⏱️ Duration (begin, peak and end time) of flare
                        \n 📑 Class type
                        \n 🗺️ Source Location
                        \n 📈 Intensity
                        """)

with tab4:
    st.title("")
    st.write("""Besides collecting the data, our small group of Stargazers (we also go for Cosmic Devotees or Galactic Fans (but never nerds)) also works on exploring the data. 
             \n We build cool plots to explore how values we collect fluctuate through time and possibly identify trends and patterns, we compare and try to find relationships between 
             different features of our data and we even dip our toes on forecasting some of these events.
             \n This is what you get when you combine three Data Scientists by day and Galactic Enthusiasts by night!
             \n The best ones by type can be found in the navigation bar's respective link ↗️

             \n We invite you to explore the datasets and use it to also conduct your own analysis and exploration. Perhaps you'll be responsiblle for the next 
             great scientific breakthrough (and we would love to know that we brought you the data to help you focus on everything else!)
             """)