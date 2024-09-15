import streamlit as st
import numpy as np
from ultralytics import YOLO

model = YOLO('best.pt')

def predict(image):
    result = model.predict(source=image, imgsz=640, conf=0.25)
    annotated_img = result[0].plot()
    return annotated_img[:, :, ::-1]

st.set_page_config(page_title="Breast Cancer Detection using YOLOv10")

st.title("Breast Cancer Detection using YOLOv10")
st.write("Upload an image to detect Breast cancer using YOLOv10")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    detected_image = predict(image)
    st.image(detected_image, use_column_width=True)
