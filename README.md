# ISL-Recognition-System-Using-YOLOv11
This project is an AI-based **image classification system** that recognizes **Indian Sign Language (ISL) gestures from images** using **YOLOv11**.  
A live demo is available on **Hugging Face Spaces**, where users can upload an image to identify the corresponding ISL alphabet.
🔗 **Live Demo:** [Hugging Face Space – SweetSitara/Project](https://huggingface.co/spaces/SweetSitara/Project)
## 📘 Overview

Sign language is essential for communication for hearing and speech-impaired individuals.  
This project detects **ISL gestures from static images**.

Workflow:

1. Upload an image of a hand gesture  
2. Detect the **hand region** using **YOLOv11**  
3. Predict the corresponding ISL alphabet  
4. Display the result on the interface  

> ⚠️ **Note:** Current version works with **images only**, not real-time video.

---

## 🚀 Features

- Detects ISL alphabets from uploaded images  
- Accurate hand region detection with **YOLOv11**  
- Simple web interface using Gradio  
- End-to-end image detection and classification  
- Easy deployment on Hugging Face Spaces  
