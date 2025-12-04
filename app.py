from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import base64

# 1. Initialize FastAPI app
app = FastAPI()

# 2. Load the Model ONCE (it's slow, so do it at startup)
try:
    # Replace 'best.pt' with the correct path to your model file
    model = YOLO('best.pt') 
except Exception as e:
    # Handle cases where the model file is not found
    print(f"Error loading model: {e}")
    model = None 

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # 3. Read the uploaded image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # 4. Run Inference (Prediction)
    # Setting 'save=False' prevents saving the output image to disk
    results = model.predict(source=image, save=False, imgsz=640, conf=0.25)

    # 5. Process and Prepare Results
    # To display bounding boxes in the frontend, it's easiest to
    # re-encode the *annotated* image and send it back.

    # Get the annotated image (where bounding boxes are drawn)
    im_array = results[0].plot()  # numpy array of the image with boxes
    im = Image.fromarray(im_array[..., ::-1])  # Convert BGR (OpenCV format) to RGB

    # Save annotated image to a bytes buffer
    buffered = BytesIO()
    im.save(buffered, format="PNG")

    # Encode as Base64 for easy transmission via JSON
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"status": "success", "annotated_image": img_str}

# Run this file with: uvicorn app:app --reload