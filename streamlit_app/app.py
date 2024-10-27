import pandas as pd
import numpy as np

import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas
from streamlit_gsheets import GSheetsConnection
from datetime import datetime


from PIL import Image, ImageDraw
import base64
import io
import os

from sidebar_utils import display_calendar_in_sidebar
from operator import add, sub, mul, truediv

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from huggingface_hub import try_to_load_from_cache, HfFolder

import pytz
import cv2

import re
import shutil
import logging
import warnings
warnings.filterwarnings('ignore')


#_______ Page Setup _______ #
st.set_page_config(
    page_title="OCR Math",
    page_icon=":1234:",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


#background_image = "images/background.jpg"
#encoded_image = get_base64_of_bin_file(background_image)


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


st.cache_resource.clear()


st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)


#_______ Paths _______ #
save_directory = "C:/Users/joana/OneDrive/Desktop/HSLU/3rd_semester/CV/c_vision_ocr/code/trocr-large-stage1-local"

#_______ Sidebar Setup _______ #
display_calendar_in_sidebar()



#_______ Functions _______ #
# cache_dir = HfFolder.cache_home
# model_id = "microsoft/trocr-small-stage1"
# shutil.rmtree(os.path.join(cache_dir, "models--" + model_id.replace("/", "--")), ignore_errors=True)


model, processor = None, None

@st.cache_data
# Loads the model without using a locally saved file
def load_model():
    try:
        with st.spinner("Loading model... This may take a few seconds."):
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten", ignore_mismatched_sizes=True)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logging.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None

# Loads the model from a locally saved file
# def load_model():
#     with st.spinner("Loading model... This may take a few minutes."):
#         processor = TrOCRProcessor.from_pretrained(save_directory)
#         model = VisionEncoderDecoderModel.from_pretrained(save_directory)
#     return processor, model


# Processes the image and creates the bounding boxes around each character
def predict_text_with_boxes(image):
    # Load model and processor
    processor, model = load_model()
    
    # Converts the PIL image to RGB 
    image_rgb = image.convert('RGB')
    
    # Preprocesses the image and gets the text prediction from TrOCR
    pixel_values = processor(image_rgb, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Converts the PIL image to OpenCV format (numpy array) for further processing
    image_np = np.array(image_rgb)
    
    # Converts the image to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Applies binary thresholding to get a binary image
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Detects the contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Empty list to store bounding boxes
    bounding_boxes = []
    
    # Loop over each contour and gets the bounding box coordinates for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    
    return generated_text, bounding_boxes

# Function to draw the bounding boxes around the characters in the provided image
def draw_bounding_boxes(image, boxes):
    # Creates an ImageDraw object to draw rectangles on the image
    draw = ImageDraw.Draw(image)
    
    # Loops over each bounding box and draws the rectangle
    for (x, y, w, h) in boxes:
        draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
    return image



def evaluate_equation(equation):
    equation = equation.replace(' ', '').replace('x', '*').replace('X', '*')
    
    def parse_expression(expression):
        # Updated regex to recognize decimal numbers as well as integers
        tokens = re.findall(r'\d*\.\d+|\d+|\+|\-|\*|\/|\(|\)', expression)
        output_queue = []
        operator_stack = []
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        
        for token in tokens:
            if re.match(r'\d*\.\d+|\d+', token):  # Check if token is a number
                output_queue.append(float(token))
            elif token in precedence:
                while (operator_stack and operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1], 0) >= precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == '(':
                    operator_stack.pop()
                else:
                    return "Invalid Equation"
        
        while operator_stack:
            if operator_stack[-1] == '(':
                return "Invalid Equation"
            output_queue.append(operator_stack.pop())
        
        return output_queue
    
    def evaluate_rpn(tokens):
        stack = []
        operators = {'+': add, '-': sub, '*': mul, '/': truediv}
        
        for token in tokens:
            if isinstance(token, float):
                stack.append(token)
            elif token in operators:
                if len(stack) < 2:
                    return "Invalid Equation"
                b, a = stack.pop(), stack.pop()
                try:
                    stack.append(operators[token](a, b))
                except ZeroDivisionError:
                    return "Invalid Equation"
        
        if len(stack) != 1:
            return "Invalid Equation"
        return stack[0]
    
    try:
        rpn = parse_expression(equation)
        if isinstance(rpn, str):
            return rpn
        result = evaluate_rpn(rpn)
        return result if isinstance(result, str) else round(result, 6)
    except Exception:
        return "Invalid Equation"
    


def save_error_report(original_text, corrected_text, conn):
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        # Read the existing data from the error_report worksheet
        error_report_df = conn.read(worksheet="error_report")

        # Create a new row
        new_row = pd.DataFrame([{
            "timestamp": timestamp,
            "original_text": original_text,
            "corrected_text": corrected_text
        }])

        # Append the new row to the existing DataFrame
        updated_error_report_df = pd.concat([error_report_df, new_row], ignore_index=True)
        #st.dataframe(updated_error_report_df)
        # Write the updated DataFrame back to the Google Sheets worksheet
        conn.update(worksheet="error_report", data=updated_error_report_df)

        st.success("Thank you for your feedback! We've recorded the error in Google Sheets.")
    except Exception as e:
        st.error(f"Failed to save error report: {e}")




def main():
    st.title("Mathematical Expression OCR")

    if 'input_method' not in st.session_state:
        st.session_state.input_method = "Upload Image"
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    col1, col2, col3, col4, col5 = st.columns([3, 1, 3, 1, 2])

    with col1:
        st.header("Input Method")
        input_method = st.radio("Choose input method:", ("Upload Image", "Draw Expression", "Capture from Camera"), key="input_method")
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                st.session_state.current_image = Image.open(uploaded_file).convert("RGB")
                # Display original uploaded image
                st.image(st.session_state.current_image, caption="Uploaded Image", use_column_width=True)


        elif input_method == "Draw Expression":
            st.image("streamlit_app/canvas_inst.png")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#000000",
                background_color="rgba(232, 234, 237, 0.4)",
                height=350,
                width=600,
                drawing_mode="freedraw",
                key="canvas",
                update_streamlit=False
            )
            

            if canvas_result.image_data is not None:
                img_array = canvas_result.image_data.astype(np.uint8)
                img = Image.fromarray(img_array).convert('RGB')
                st.session_state.current_image = img.copy()

        # Camera input method
        elif input_method == "Capture from Camera":
            camera_image = st.camera_input("Once the expression is in view, click Take Photo")
            if camera_image is not None:
                st.session_state.current_image = Image.open(camera_image).convert("RGB")
                st.image(st.session_state.current_image, caption="Camera Capture", use_column_width=True)

    with col3:
        st.header("Process Image")
        if st.button("Extract and Calculate"):
            if st.session_state.current_image is not None:
                try:
                    extracted_text, bounding_boxes = predict_text_with_boxes(st.session_state.current_image)
                    result_image = draw_bounding_boxes(st.session_state.current_image, bounding_boxes)
                    st.session_state.extracted_text = extracted_text
                    
                    # Display extracted text
                    st.write("Extracted Text:")
                    st.write(extracted_text)

                  
                    # Calculate result from extracted text
                    result = evaluate_equation(extracted_text)
                    st.write("Calculation Result:")
                    if isinstance(result, str):
                        st.write(result)
                    else:
                        st.write(f"{extracted_text} = {result}")
                    with st.expander("Show Image with Bounding Boxes"):
                        # Display the image with bounding boxes when the checkbox is checked
                        st.image(result_image, caption="Image with Bounding Boxes", use_column_width=True)

                except ValueError as ve:
                    st.error(f"Error processing image: {str(ve)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.error("Please upload or draw an image first.")


        with col4:
            st.write("")

        with col5:
            st.header("Help Us Improve")
            
            if st.button("Report Error"):
                st.session_state.show_error_form = True

            if 'show_error_form' in st.session_state and st.session_state.show_error_form:
                st.write("Please enter the correct text for the image:")
                
                corrected_text = st.text_input(
                    "Correct text:", 
                    value=st.session_state.get('extracted_text', ''),
                    key="corrected_text_input"
                )
                
                if st.button("Submit correction"):
                    if 'extracted_text' in st.session_state:
                        # Call the Google Sheets saving function
                        save_error_report(
                            original_text=st.session_state.get('extracted_text', 'No model text available'),
                            corrected_text=corrected_text,
                            conn=conn  # Use the Google Sheets connection
                        )
                        st.session_state.show_error_form = False
                    else:
                        st.error("No extracted text available. Please process an image first.")


            # Close the connection when the app stops
            #connection.close()


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    position: relative;
    z-index: 1;
}}

[data-testid="stAppViewContainer"] > .main::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("https://images.unsplash.com/photo-1531346878377-a5be20888e57?q=80&w=1968&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
    opacity: 0.6; /* Adjust opacity here */
    z-index: -1;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)




if __name__ == "__main__":
    main()