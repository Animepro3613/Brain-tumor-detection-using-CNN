import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("brain_tumor_detection_model.h5")

# Function to make predictions
def predict(image):
    # Preprocess the image
    img = cv2.resize(image, (240, 240))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    return prediction[0][0]

# Streamlit App
st.title("Brain Tumor Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make predictions when a button is clicked
    if st.button("Predict"):
        prediction_score = predict(image)

        # Display the prediction result
        st.subheader("Prediction Result:")
        st.write(f"Prediction Score: {prediction_score}")
        st.write(f"Class: {'Yes' if prediction_score > 0.5 else 'No'}")
