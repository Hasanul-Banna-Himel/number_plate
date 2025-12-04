import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

# Use Streamlit's caching decorator to load the heavy model only once
@st.cache_resource
def load_yolo_model():
    """Load the YOLO model weights from best.pt."""
    # Ensure 'best.pt' is in the root of your GitHub repository
    # The 'ultralytics' library automatically handles the model path
    try:
        model = YOLO('best.pt') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main App Logic ---

# 1. Page Setup
st.title("YOLO Number Plate Detection App")
st.markdown("Upload an image to detect number plates using the custom `best.pt` model.")

# Load the model
model = load_yolo_model()

if model is None:
    st.stop() # Stop the app if the model failed to load

# 2. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3. Read and Display Original Image
    st.header("Original Image")
    
    # Open the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Image uploaded successfully.', use_column_width=True)
    
    # 4. Run Detection on Button Click
    if st.button("Run Detection"):
        with st.spinner('Running object detection...'):
            
            # Convert PIL image to numpy array for YOLO prediction
            img_np = np.array(image)
            
            # Run Inference
            # The 'source' can be a PIL Image or numpy array
            results = model.predict(source=img_np, save=False, imgsz=640, conf=0.25)
            
            # 5. Process and Display Results
            
            # Use the .plot() method to get the image with bounding boxes drawn
            annotated_img_array = results[0].plot() 
            
            # YOLO's .plot() uses BGR format (OpenCV default), so convert back to RGB for Streamlit
            annotated_img_rgb = cv2.cvtColor(annotated_img_array, cv2.COLOR_BGR2RGB)
            
            st.header("Detection Results")
            st.image(annotated_img_rgb, caption='Detected Number Plates', use_column_width=True)
            
            # Optional: Display raw detection data
            st.subheader("Raw Results Data")
            st.dataframe(results[0].pandas().xyxy[0])
