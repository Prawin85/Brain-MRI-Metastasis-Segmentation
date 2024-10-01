# Brain-MRI-Metastasis-Segmentation
mkdir brain_mri_segmentation
cd brain_mri_segmentation
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install numpy pandas pydicom opencv-python albumentations tensorflow fastapi uvicorn streamlit
import zipfile
import requests
import os

url = "https://dicom5c.blob.core.windows.net/public/Data.zip"
response = requests.get(url)
with open("Data.zip", "wb") as f:
    f.write(response.content)

with zipfile.ZipFile("Data.zip", 'r') as zip_ref:
    zip_ref.extractall("Data")
import cv2

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)
import albumentations as A

augmentation = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.Normalize()
])
from tensorflow.keras import layers, models

def nested_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    # Build your Nested U-Net architecture here
    return models.Model(inputs, outputs)
def attention_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    # Build your Attention U-Net architecture here
    return models.Model(inputs, outputs)
model_nu = nested_unet(input_shape=(256, 256, 1))
model_nu.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_au = attention_unet(input_shape=(256, 256, 1))
model_au.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_nu = model_nu.fit(train_data, train_labels, validation_split=0.2, epochs=50)
model_nu.save("nested_unet_weights.h5")

history_au = model_au.fit(train_data, train_labels, validation_split=0.2, epochs=50)
model_au.save("attention_unet_weights.h5")
def dice_score(y_true, y_pred):
    smooth = 1e-6
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

# Evaluate and compare DICE scores
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and process image, then return prediction
    return JSONResponse(content={"result": "segmentation mask"})
import streamlit as st

st.title("Brain MRI Metastasis Segmentation")
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "dicom"])
if uploaded_file:
    # Process the uploaded file and display results
    st.image("segmentation_mask.png", caption="Segmentation Result")
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin master
