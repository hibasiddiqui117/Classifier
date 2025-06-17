import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('Image_classify.keras')

# Class names (adjust manually if needed)
class_names = ['apple', 'banana', 'cabbage', 'carrot']  # Change as per your folders

# App UI
st.title("🍎🥦 Image Classifier: Fruits vs Vegetables")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # batch of 1

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]

    st.success(f"Prediction: {predicted_class} ({100 * np.max(score):.2f}% confidence)")
