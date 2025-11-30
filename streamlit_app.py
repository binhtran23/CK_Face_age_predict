import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow import keras

Width_Min, Height_Min = 224, 224

# Load the age prediction model
@st.cache(allow_output_mutation=True)
def load_model():
    return keras.models.load_model("best_model_age.h5")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    #Get width and height
    h, w = img_array.shape[:2]
    if h < Height_Min or w < Width_Min:
        raise ValueError(f"Image size is too small. Minimum size is {Width_Min}x{Height_Min}px.")
    # Ensure RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    face_casscade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    assert not face_casscade.empty(), "Failed to load haarcascade classifier xml file."
    
    faces = face_casscade.detectMultiScale(img_array, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    #Only use the first detected face for prediction
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img_array = img_array[y:y+h, x:x+w]
    
    # Resize to model input size (224x224)
    img_resized = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Convert to float32 and normalize to [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension for model input
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_age(model, image):
    """Predict age from image using the loaded model"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    
    # Model outputs single age value for regression
    predicted_age = float(prediction[0][0])
    
    # Ensure reasonable age range (clip to 0-100)
    predicted_age = max(0, min(100, predicted_age))
    
    return predicted_age

st.title("Mô hình dự đoán tuổi khuôn mặt")

# Load the model
model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Make prediction
    with st.spinner("Đang dự đoán tuổi..."):
        predicted_age = predict_age(model, image)

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Ảnh mặt phát hiện")
        img_array = np.array(image)
        face_casscade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_casscade.detectMultiScale(img_array, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 0, 255), 2)
        st.image(img_array, use_column_width=True)    

    with col3:
        st.subheader("Kết quả dự đoán")
        st.metric("Tuổi dự đoán", f"{predicted_age:.1f} tuổi")
        
        # Display additional info
        st.info(f"Mô hình dự đoán tuổi của người trong ảnh là: **{predicted_age:.1f} tuổi**")
else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu dự đoán tuổi.")
