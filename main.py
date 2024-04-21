# import library
import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os

# Setting color scheme and styles
background_color = "#0B0C10"
text_color = "#C5C6C7"
button_color = "#66FCF1"
button_hover_color = "#45A29E"
header_color = "#8b8c8c"
font_family = "Arial, sans-serif"

# custom setting for website
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: {font_family};
    }}
    .stButton>button {{
        background-color: {button_color};
        color: {background_color};
        border-radius: 4px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        padding: 10px 24px;
        transition: background-color 0.3s, box-shadow 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {button_hover_color};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {header_color};
    }}
    .st-cj {{
        background-color: rgba(31, 40, 51, 0.8);  // Adjust the alpha for transparency
        padding: 10px;
        border-radius: 5px;
    }}
    .css-145kmo2 {{
        background-color: {header_color};
        color: {text_color};
    }}
    .st-af {{
        margin: 20px 0;  // Adds spacing around the nav bar
    }}
    </style>
    """,
    unsafe_allow_html=True
)




#model path
model_path = r"C:\Users\diwak\OneDrive\Desktop\AGT assesment\runs (1)\detect\train\weights\last.pt"

#import trained model
model = YOLO(model_path)

#heading
st.header("App/Web Element Detections")
uploaded_files = st.file_uploader(
        "Click on Browse file and choose an image", accept_multiple_files=True)

# Enter a some space
st.markdown("<br>", unsafe_allow_html=True)



if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            #create a temporary file which is uploaded by user
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            #read that file
            tfile.write(uploaded_file.read())

            # label and color for each class
            classNames = ["text", "button", "radio button","checkbox","input field","navigation dot"]
            colors = [(0, 255, 0),(0, 255, 0),(0, 255, 0),(0, 255, 0),(0, 255, 0),(0, 255, 0) ]

            # open image
            image = cv2.imread(tfile.name)
            #predict elements using trained model
            res = model.predict(image,save=True)
        
            
            #show the result
            img_path = r"runs\detect\predict\image0.jpg"
            st.image(img_path)
            

