import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras

Width_Min, Height_Min = 224, 224

# Load YuNet face detector
@st.cache(allow_output_mutation=True)
def load_face_detector():
    model_path = 'face_detection_yunet_2023mar.onnx'
    if not os.path.exists(model_path):
        st.error(f"YuNet model not found: {model_path}")
        return None
    
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3
    )
    return face_detector

# Load the age prediction model
@st.cache(allow_output_mutation=True)
def load_model():
    return keras.models.load_model("best_model_age.h5")

def detect_and_crop_face(img_array, target_size=(224, 224), margin=0.3):
    """
    Detect and crop face using YuNet detector.
    Same preprocessing as in preprocess.ipynb
    """
    face_detector = load_face_detector()
    if face_detector is None:
        raise ValueError("Face detector not loaded")
    
    h_img, w_img = img_array.shape[:2]
    
    # YuNet Setup
    face_detector.setInputSize((w_img, h_img))
    _, faces = face_detector.detect(img_array)
    
    # Only process images with EXACTLY 1 face
    if faces is None or len(faces) != 1:
        return None, None
    
    # Get face coordinates
    face = faces[0]
    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    
    # Minimum face size filter
    MIN_FACE_SIZE = 64
    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return None, None
    
    # Square crop with center alignment
    center_x = x + w // 2
    center_y = y + h // 2
    
    max_dim = max(w, h)
    side_length = int(max_dim * (1 + margin))
    
    x1 = center_x - side_length // 2
    y1 = center_y - side_length // 2
    x2 = x1 + side_length
    y2 = y1 + side_length
    
    # Handle borders with padding
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_img)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_img)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        img_array = cv2.copyMakeBorder(img_array, pad_top, pad_bottom, pad_left, pad_right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top
    
    # Crop square region
    cropped = img_array[y1:y2, x1:x2]
    
    # Resize to target size
    cropped_resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    # Return cropped face and bounding box info for visualization
    bbox = (x, y, w, h)
    return cropped_resized, bbox

def preprocess_image(image):
    """Preprocess image for model prediction - matching preprocess.ipynb pipeline"""
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image)
    
    # Ensure RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Detect and crop face using YuNet (same as preprocess.ipynb)
    cropped_face, bbox = detect_and_crop_face(img_array, target_size=(224, 224), margin=0.3)
    
    if cropped_face is None:
        raise ValueError("Không phát hiện được khuôn mặt hoặc phát hiện nhiều hơn 1 khuôn mặt. Vui lòng sử dụng ảnh có đúng 1 khuôn mặt.")
    
    # Convert to float32 and normalize to [0,1] (matching training pipeline)
    img_normalized = cropped_face.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, bbox

def predict_age(model, image):
    """Predict age from image using the loaded model"""
    try:
        processed_image, bbox = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0).round()
        
        # Model outputs single age value for regression
        predicted_age = float(prediction[0][0])
        
        # Ensure reasonable age range (clip to 0-100)
        predicted_age = max(0, min(100, predicted_age))
        
        return predicted_age, bbox
    except ValueError as e:
        raise e

st.title("Mô hình dự đoán tuổi khuôn mặt")
st.markdown("Sử dụng YuNet Face Detector và CNN Regression Model")

# Load the model
model = load_model()

uploaded_file = st.file_uploader("Tải lên ảnh khuôn mặt", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Make prediction
    try:
        with st.spinner("Đang phát hiện khuôn mặt và dự đoán tuổi..."):
            predicted_age, bbox = predict_age(model, image)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Kết quả dự đoán")
            st.metric("Tuổi dự đoán", f"{predicted_age:.0f} tuổi", delta=None)
            

        # Success message
        st.success(f"Dự đoán hoàn tất!")
            
    except ValueError as e:
        st.error(f"Lỗi: {str(e)}")
        st.info("Lưu ý: Ảnh phải có đúng 1 khuôn mặt rõ ràng để dự đoán chính xác.")
        
else:
    st.info("Vui lòng tải lên một ảnh để bắt đầu dự đoán tuổi.")