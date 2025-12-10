import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from efficientnet_pytorch import EfficientNet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #2E1F1A 0%, #241713 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #F5E9E2;
}

/* Smooth fade-in animation */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* Main content box */
.main-box {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    padding: 40px;
    border-radius: 25px;
    width: 85%;
    margin: auto;
    margin-top: 40px;
    animation: fadeIn 1s ease-in-out;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

/* Title */
h1 {
    text-align: center;
    color: #F5E9E2 !important;
    font-weight: 900;
    letter-spacing: 2px;
    text-shadow: 0 0 25px rgba(255, 200, 150, 0.35);
    animation: fadeIn 1.2s ease;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #EED6C4;
    font-size: 19px;
    margin-bottom: 20px;
    opacity: 0.9;
}

/* Upload box */
.stFileUploader {
    background: rgba(255, 255, 255, 0.09) !important;
    padding: 14px;
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    box-shadow: inset 0 0 12px rgba(255,255,255,0.12);
    transition: 0.3s;
}
.stFileUploader:hover {
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(255,200,160, 0.25);
}

/* Skin Type Badge */
.skin-badge {
    background: linear-gradient(135deg, #C8A38D 0%, #e8c2aa 100%);
    color: #3A2A22;
    padding: 10px 20px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    display: inline-block;
    box-shadow: 0 0 15px rgba(200,150,120, 0.35);
    animation: fadeIn 1.3s ease;
}

/* Confidence Box */
.confidence-box {
    background: rgba(255,255,255,0.08);
    padding: 12px;
    border-radius: 10px;
    margin: 7px 0;
    border-left: 4px solid #C8A38D;
    color: #F7EBE4;
    backdrop-filter: blur(6px);
    animation: fadeIn 1.2s ease;
}

/* Product Card */
.product-card {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
    border-left: 6px solid #D6B1A2;
    border-top: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 3px 15px rgba(0,0,0,0.35);
    animation: fadeIn 1.4s ease;
    transition: 0.3s ease;
}
.product-card:hover {
    transform: scale(1.025) translateY(-4px);
    box-shadow: 0 8px 25px rgba(255,180,150,0.25);
    border-left: 6px solid #E3C0B3;
}

/* Product title inside card */
.product-card h3 {
    color: #FFE9DC;
    text-shadow: 0 0 12px rgba(255,200,150,0.25);
}

/* Divider customization */
hr {
    border-color: rgba(255,255,255,0.15) !important;
}

</style>
""", unsafe_allow_html=True)


# HEADER
st.markdown("<h1>AI Skin Type Detector & Beauty Recommender</h1>", unsafe_allow_html=True)
st.write("Upload your face photo to detect your skin type and get smart product recommendations.")


# LOAD MODEL
@st.cache_resource
def load_model():
    model = EfficientNet.from_name("efficientnet-b0")
    num_features = model._fc.in_features
    model._fc = nn.Linear(num_features, 4)
    model.load_state_dict(torch.load("../models/efficientnet_skin_classifier.pth",
                                     map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
class_names = ["acne-prone-skin", "dry-skin", "oily-skin", "healthy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_skin_type(img_pil):
    img_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    return class_names[pred_idx], probs.numpy()


# LOAD PRODUCT DATA + RECOMMENDER
@st.cache_resource
def load_data():
    df = pd.read_csv("/Users/Admin/Desktop/project/cosmetics_full_data.csv")

    def extract_number(x):
        if isinstance(x, str):
            num = ''.join([c for c in x if c.isdigit() or c == '.'])
            return float(num) if num else np.nan
        return np.nan

    df["Product_Size"] = df["Product_Size"].apply(extract_number)
    df = df.drop_duplicates()

    num_cols = ["Price_USD", "Rating", "Number_of_Reviews", "Product_Size"]
    for col in num_cols:
        df[col] = df[col].astype(float).fillna(df[col].median())

    df["Skin_Type"] = df["Skin_Type"].fillna("Unknown")
    df["Main_Ingredient"] = df["Main_Ingredient"].fillna("")

    df["Cruelty_Score"] = df["Cruelty_Free"].apply(lambda x: 1 if str(x).lower() == "yes" else 0)
    df["Price_Score"] = (df["Price_USD"].max() - df["Price_USD"]) / df["Price_USD"].max()

    tfidf = TfidfVectorizer(stop_words="english")
    ingredient_matrix = tfidf.fit_transform(df["Main_Ingredient"])

    return df, ingredient_matrix

df, ingredient_matrix = load_data()


def recommend_products_optimized(skin_type, top_n=10):
    user_skin = skin_type.lower()
    df["Skin_Match_Score"] = df["Skin_Type"].apply(lambda x: 1 if str(x).lower() in user_skin else 0)

    skin_products = df[df["Skin_Type"].str.lower().str.contains(user_skin)]

    if len(skin_products) > 0:
        avg_vector = ingredient_matrix[skin_products.index].mean(axis=0)
        df["Ingredient_Similarity"] = cosine_similarity(ingredient_matrix, avg_vector)
    else:
        df["Ingredient_Similarity"] = 0

    df["Review_Score"] = np.log1p(df["Number_of_Reviews"])

    df["Popularity_Score"] = (
        df["Rating"] * 0.5 +
        df["Review_Score"] * 0.3 +
        df["Cruelty_Score"] * 0.1
    )

    df["Final_Score"] = (
        df["Popularity_Score"] * 0.5 +
        df["Ingredient_Similarity"] * 0.3 +
        df["Skin_Match_Score"] * 0.2
    )

    return df.sort_values("Final_Score", ascending=False).head(top_n)


# UPLOAD IMAGE
uploaded_img = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])

if uploaded_img:
    img_pil = Image.open(uploaded_img).convert("RGB")

    st.image(img_pil, caption="Your Uploaded Image", width=300)

    st.subheader("🔍 Skin Type Detection")
    
    skin_type, probs = predict_skin_type(img_pil)

    st.markdown(f"<div class='skin-badge'>Predicted Skin Type: <b>{skin_type.upper()}</b></div>", unsafe_allow_html=True)

    st.write("### Confidence Levels")
    for i, cls in enumerate(class_names):
        st.markdown(f"<div class='confidence-box'><b>{cls}</b>: {probs[i]:.4f}</div>", unsafe_allow_html=True)

    st.divider()

    # PRODUCT RECOMMENDATIONS
    st.markdown("## ✨ Recommended Products Just For You")

    recommended = recommend_products_optimized(skin_type)

    for _, row in recommended.iterrows():
        st.markdown(f"""
        <div class="product-card">
            <h3>{row['Product_Name']}</h3>
            <b>Brand:</b> {row['Brand']} <br>
            <b>Category:</b> {row['Category']} <br>
            <b>Price:</b> 💵 ${row['Price_USD']} <br>
            <b>Rating:</b> ⭐ {row['Rating']} <br>
            <b>Reviews:</b> {row['Number_of_Reviews']} <br>
            <b>Main Ingredient:</b> {row['Main_Ingredient']} <br>
        </div>
        """, unsafe_allow_html=True)
