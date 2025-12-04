import gradio as gr
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt') 

def predict_image(img):
    """Runs the model on the input image and returns the annotated image."""
    # 'img' is a numpy array provided by Gradio
    results = model.predict(source=img, save=False, imgsz=640, conf=0.25)

    # .plot() returns a numpy array with bounding boxes drawn
    annotated_img_array = results[0].plot() 
    return annotated_img_array

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detection Results"),
    title="YOLO Object Detection Demo",
    description="Upload an image to detect objects using the 'best.pt' model."
)

# Launch the app
iface.launch()
