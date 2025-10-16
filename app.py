import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLO model
model = YOLO("best.pt")  # Replace with "yolov8n.pt" or "yolov11.pt" if needed

def detect(image):
    # Save the uploaded image temporarily
    image.save("temp.jpg")
    
    # Perform object detection
    results = model("temp.jpg")
    
    # Get annotated result (BGR NumPy array)
    annotated_bgr = results[0].plot()
    
    # Convert BGR to RGB for correct color display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    # Return as PIL image
    return Image.fromarray(annotated_rgb)

# Create Gradio interface
interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Indian Sign Language Recognition"
)

# Launch app
interface.launch()
