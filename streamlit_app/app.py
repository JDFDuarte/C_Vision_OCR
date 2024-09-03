#_______ Library Import _______ #
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from datetime import datetime
import numpy as np
import json
import base64
from PIL import Image
import io
import os
from io import StringIO
from ocr_test_run import recognize_characters
from ocr_test_run import process_math_image
from ocr_test_run import parse_expression
from ocr_test_run import safe_eval
import easyocr 
#import markdown_functions as md

#_______ Page Setup _______ #
st.set_page_config(
    page_title="NASA Weather Exploration",
    page_icon=":1234:",
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

#st.markdown(md.get_navbar_markdown(), unsafe_allow_html=True)
#st.markdown(md.background_and_tabs_styles(encoded_image), unsafe_allow_html = True)


# _______ Page _______ #
st.write("")
st.title("")
col_1, col_2, col_3 = st.columns([7, 2, 1])

with col_1:
    st.title("")
    st.title("**Calculator**")
    #st.subheader("A Cross-Planetary Study of Earth's and Mars's Atmospheric Patterns")

with col_2:
    st.title("")
    #st.image("images/logo_cs.png", width=350)


container = st.container(border=True)

container.write("Calculator")


selection = st.selectbox(
    "How would you like to upload an image?",
    ("Take a Picture", "Upload Image")
)


reader = easyocr.Reader(['en'])


picture = None
if selection == "Take a Picture":
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        picture = st.camera_input("Take a picture")
elif selection == "Upload Image":
    picture = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

if picture is not None:
    image = Image.open(picture)
    col1, col2, col3 = st.columns([1.5, 1, 1.5])  
    with col2:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image_np = np.array(image)
    result = reader.readtext(image_np)
    
    recognized_text = ' '.join([text for _, text, _ in result])
    st.write("Recognized Text:")
    st.write(recognized_text)
    
    try:
        expression = parse_expression(recognized_text)
        result = safe_eval(expression)
        st.write(f"Expression: {expression}")
        st.write(f"Result: {result}")
    except Exception as e:
        st.write("Unable to evaluate the expression")
        st.write(f"Error: {e}")

st.write("Calculator")


