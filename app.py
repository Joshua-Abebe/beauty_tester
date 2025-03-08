
import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load your trained model
learn_inf = load_learner('export.pkl')

# Streamlit UI
st.title("Lady Classifier App 👩‍🎨")  # App title

st.write("Upload an image to classify:")

# Upload Image Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Convert to FastAI format
    img_fastai = PILImage.create(uploaded_file)

    # Predict
    pred, pred_idx, prob = learn_inf.predict(img_fastai)

    # Display result
    st.markdown(f"### **Prediction:** {pred}")
    st.markdown(f"### **Probability:** {prob[pred_idx]:.4f}")
