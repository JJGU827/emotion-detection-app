import os
import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Get the absolute path to the directory the script is in
base_dir = os.path.dirname(os.path.abspath(__file__))

# Emotion map
emotion_map = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(os.path.join(base_dir, "resnet50_fer2013.pth"), map_location=device))
model.to(device)
model.eval()

# Face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Transform for input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_emotion_from_image(image: np.ndarray) -> tuple[str, float] | None:
    """
    Detects face in an image and predicts the emotion of the first face found.

    Args:
        image (np.ndarray): BGR image (as from cv2.imread or cv2.VideoCapture)

    Returns:
        Tuple of (emotion_label: str, confidence: float) if face found, else None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("[ERROR] No face detected.")
        return None

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output, dim=1).item()
            emotion = emotion_map[pred_class]
            confidence = F.softmax(output, dim=1)[0][pred_class].item() * 100

        return emotion, confidence  # Only predict first face


def app():
    st.set_page_config(
        page_title="Ex-stream-ly Cool App",
        page_icon="🚗"
    )
    st.header("Distracted Driver App", divider="gray")
    st.markdown("Project by Daivien, Anish, Sienna, Chad, Harris, Angelina, Yusef")

    picture = st.camera_input("Take a picture")
    if picture:
        image = np.array(Image.open(picture).convert("RGB"))
        results = predict_emotion_from_image(image)
        if results:
            emotion, score = results
            st.markdown(f"Emotion: {emotion}")
            st.markdown(f"Model Confidence: {score}")
        else:
            st.markdown("NO FACE DETECTED")

if __name__ == "__main__":
    app()
