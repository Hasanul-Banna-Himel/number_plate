import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# ---------------------------
# Load YOLO model
# ---------------------------
model = YOLO("best.pt")   
# ---------------------------
# Prediction function
# ---------------------------
def predict_image(uploaded_image):
    # Convert uploaded file to PIL Image → numpy array
    img = Image.open(uploaded_image).convert("RGB")
    img_array = np.array(img)

    # Run YOLO inference
    results = model.predict(source=img_array, save=False, imgsz=640, conf=0.25)

    # Draw bounding boxes
    annotated_img = results[0].plot()

    return annotated_img


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("📦 YOLO Object Detection Demo")
st.write("Upload an image to detect objects using **best.pt** YOLO model.")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# When image is uploaded → run detection
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Detecting... Please wait"):
            result_image = predict_image(uploaded_image)

        # Convert numpy → displayable image
        st.image(result_image, caption="Detection Results", use_column_width=True)

