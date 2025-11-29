import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
from tensorflow import keras

# Load the age prediction model
@st.cache(allow_output_mutation=True)
def load_model():
    return keras.models.load_model("best_model_age.h5")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if image has alpha channel
    if img_array.shape[-1] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Resize image to expected input size (assuming 224x224, adjust if needed)
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def predict_age(model, image):
    """Predict age from image using the loaded model"""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Assuming the model outputs a single value for age
    predicted_age = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
    
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
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh gốc")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Kết quả dự đoán")
        st.metric("Tuổi dự đoán", f"{predicted_age:.1f} tuổi")
        
        # Display additional info
        st.info(f"Mô hình dự đoán tuổi của người trong ảnh là: **{predicted_age:.1f} tuổi**")
else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu dự đoán tuổi.")

# Display model info
st.sidebar.subheader("Thông tin mô hình")
st.sidebar.info("Sử dụng mô hình: best_model_age.h5")
st.sidebar.info("Mô hình được huấn luyện để dự đoán tuổi từ ảnh khuôn mặt")