import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Skin Type AI Recommender",
    page_icon="💄",
    layout="centered"
)

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "..", "models", "skin_classifier.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

class_names = ["acne-prone-skin", "dry-skin", "healthy", "oily-skin"]


# ==================================================
# LOAD DATASET (CSV)
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("../data/cosmetics_full_data.csv")

df = load_data()

# ==================================================
# HEADER
# ==================================================
st.markdown("<h1 style='text-align:center;'>💄 Skin Type AI Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Detect your skin type & get best product recommendations</p>", unsafe_allow_html=True)
st.markdown("---")

# ==================================================
# USER MODE
# ==================================================
mode = st.radio("Choose Input Method:", ["📸 AI Image Analysis", "✍️ Manual Selection"])

# ==================================================
# PREDICTION FUNCTION
# ==================================================
def predict_skin_type(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    probs = prediction[0]

    predicted_index = np.argmax(probs)
    predicted_label = class_names[predicted_index]

    return predicted_label, probs

skin_type = None

# ==================================================
# AI MODE
# ==================================================
if mode == "📸 AI Image Analysis":
    uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        skin_type, probs = predict_skin_type(image)

        # Create 2 columns
        col1, col2 = st.columns([1, 1])

        # LEFT → Image
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # RIGHT → Prediction + Confidence
        with col2:
            st.success(f"🧠 Predicted Skin Type: **{skin_type}**")

            st.markdown("### 🔍 Confidence Scores")

            for i, name in enumerate(class_names):
                st.write(name)
                st.progress(float(probs[i]))

# ==================================================
# MANUAL MODE
# ==================================================
elif mode == "✍️ Manual Selection":
    skin_type = st.selectbox("Select your skin type:", class_names)

    if skin_type:
        st.success(f"Selected Skin Type: **{skin_type}**")

# ==================================================
# SKIN TYPE MAPPING (IMPORTANT)
# ==================================================
mapping = {
    "oily-skin": "Oily",
    "dry-skin": "Dry", 
    "acne-prone-skin": "Sensitive",
    "healthy": "Normal"
}

# ==================================================
# RECOMMENDATION SECTION
# ==================================================
if skin_type:
    st.markdown("---")
    st.markdown("## 🎯 Recommended Products")

    dataset_skin = mapping.get(skin_type, skin_type)

    filtered = df[df["Skin_Type"].str.lower() == dataset_skin.lower()]

    # Sort by rating
    filtered = filtered.sort_values(by="Rating", ascending=False)

    if filtered.empty:
        st.warning("No products found for this skin type.")
    else:
        st.dataframe(
            filtered[[
                "Product_Name",
                "Brand",
                "Category",
                "Price_USD",
                "Rating",
                "Number_of_Reviews"
            ]],
            use_container_width=True
        )


# ================================
# CHARTS SECTION
# ================================
st.markdown("### 📊 Price Analysis")

st.bar_chart(
    filtered.set_index("Product_Name")["Price_USD"]
)

st.markdown("### ⭐ Rating Analysis")

st.bar_chart(
    filtered.set_index("Product_Name")["Rating"]
)


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("AI-Based Skin Analysis + Smart Recommendation System")