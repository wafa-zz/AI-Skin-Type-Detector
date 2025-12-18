import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet


# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Skin Type & Beauty Recommender",
    layout="wide"
)
st.title("AI Skin Type & Beauty Recommender")
st.markdown(
    "Select your skin type and optionally upload a face or skin image for AI-assisted analysis."
)

# User skin type input 
# --------------------------------------------------
st.subheader("Select Your Skin Type")

USER_SKIN_TYPE = st.radio(
    "This helps us give accurate recommendations",
    [
        "Dry",
        "Oily",
        "Combination",
        "Sensitive",
        "Acne-Prone",
        "Normal"
    ],
    horizontal=True
)

USER_SKIN_TYPE = USER_SKIN_TYPE.lower()


# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = EfficientNet.from_name("efficientnet-b0")
    model._fc = nn.Linear(model._fc.in_features, 4)
    model.load_state_dict(
        torch.load(
            "/Users/Admin/Desktop/project/models/efficientnet_skin_classifier.pth",
            map_location=torch.device("cpu")
        )
    )
    model.eval()
    return model

model = load_model()

class_names = ["acne-prone-skin", "dry-skin", "oily-skin", "healthy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# SKIN DETECTION (HSV fallback)
# --------------------------------------------------
def detect_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size
    return skin_ratio > 0.02

# --------------------------------------------------
# FACE / SKIN VALIDATION
# --------------------------------------------------
def is_valid_skin_image(image_bgr):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.02,
        minNeighbors=2,
        minSize=(15, 15)
    )

    if len(faces) > 0:
        return True

    return detect_skin(image_bgr)

# --------------------------------------------------
# SKIN TYPE PREDICTION (AI)
# --------------------------------------------------
def predict_skin_type(img_pil):
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if not is_valid_skin_image(img_bgr):
        return None, None, False

    img_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    idx = torch.argmax(probs).item()
    
    # Map AI prediction to user-friendly format
    ai_skin_raw = class_names[idx]
    ai_skin_mapped = ai_skin_raw.replace("-skin", "").replace("-", "-")
    
    return ai_skin_mapped, probs[idx].item() * 100, True

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("/Users/Admin/Desktop/project/data/cosmetics_full_data.csv")
    df["Skin_Type"] = df["Skin_Type"].fillna("").str.lower()
    return df

df = load_data()

# --------------------------------------------------
# SKIN TYPE MAPPING (DATASET)
# --------------------------------------------------
SKIN_TYPE_MAP = {
    "dry": ["dry"],
    "oily": ["oily"],
    "combination": ["combination", "oily"],
    "sensitive": ["sensitive"],
    "acne-prone": ["acne", "acne prone", "oily"],
    "acne": ["acne", "acne prone", "oily"],
    "normal": ["normal", "all"],
    "healthy": ["normal", "all"]
}


# --------------------------------------------------
# SIMPLE RECOMMENDER (BY SKIN TYPE)
# --------------------------------------------------
def recommend_products(skin_type, top_n=10):
    """
    Get products for a specific skin type
    """
    keywords = SKIN_TYPE_MAP.get(skin_type, [skin_type])
    
    mask = df["Skin_Type"].apply(
        lambda x: any(k in x for k in keywords)
    )
    
    filtered = df[mask]
    
    if filtered.empty:
        return pd.DataFrame()
    
    filtered = filtered.sort_values(
        ["Rating", "Number_of_Reviews"],
        ascending=False
    )
    
    return filtered[[
        "Product_Name",
        "Brand",
        "Category",
        "Price_USD",
        "Rating",
        "Number_of_Reviews",
        "Main_Ingredient",
        "Skin_Type"
    ]].head(top_n)
# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("🧾 View Options")
    show_cards = st.toggle("Show Card View", value=False)
    
    st.markdown("---")
    st.subheader("📊 Recommendation Logic")
    st.markdown("""
    - **User Selection**: Primary preference
    - **AI Detection**: Secondary guidance
    - **Hybrid Mode**: Combines both for best results
    """)

# --------------------------------------------------
# IMAGE UPLOAD (OPTIONAL)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Optional: Upload face / skin image (AI assist)",
    type=["jpg", "jpeg", "png"]
)

ai_skin = None
confidence = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=320)

    with col2:
        with st.spinner("AI analyzing image..."):
            ai_skin, confidence, valid = predict_skin_type(image)

        if valid:
            st.success(
                f"AI Detected: **{ai_skin.replace('-', ' ').title()} Skin**"
            )
            st.info(f"Confidence: {confidence:.2f}%")
            
            # Show comparison if different
            if ai_skin != USER_SKIN_TYPE:
                st.warning(
                    f"Your selection: **{USER_SKIN_TYPE.title()}**\n\n"
                    f"AI detected: **{ai_skin.title()}**\n\n"
                    "We'll combine both for better recommendations!"
                )
        else:
            st.warning(
                "⚠️ AI could not reliably detect skin. "
                "Using your selected skin type only."
            )
            ai_skin = None
            
# --------------------------------------------------
# SKIN TYPE SELECTOR (if AI detected something different)
# --------------------------------------------------
st.markdown("---")
# Determine which skin types are available
available_skin_types = [USER_SKIN_TYPE]
skin_type_labels = [f"🟢 {USER_SKIN_TYPE.title()} (Your Selection)"]
if ai_skin and ai_skin != USER_SKIN_TYPE:
    available_skin_types.append(ai_skin)
    skin_type_labels.append(f"🔵 {ai_skin.title()} (AI Detected)")
# Show selector if we have multiple options
if len(available_skin_types) > 1:
    st.subheader("Choose Skin Type for Recommendations")
    
    selected_skin_index = st.radio(
        "Select which skin type recommendations you want to see:",
        range(len(available_skin_types)),
        format_func=lambda x: skin_type_labels[x],
        horizontal=True
    )
    
    selected_skin_type = available_skin_types[selected_skin_index]
    
    # Show helpful info
    if selected_skin_index == 0:
        st.info(f"💡 Showing products for **{USER_SKIN_TYPE.title()}** skin based on your selection")
    else:
        st.info(f"💡 Showing products for **{ai_skin.title()}** skin based on AI analysis")
else:
    selected_skin_type = USER_SKIN_TYPE
    st.subheader(f"Recommended Products for **{USER_SKIN_TYPE.title()} Skin**")
    
    
# --------------------------------------------------
# GET RECOMMENDATIONS FOR SELECTED SKIN TYPE
# --------------------------------------------------
recommendations = recommend_products(selected_skin_type, top_n=10)
if recommendations.empty:
    st.warning(f"⚠️ No products found for {selected_skin_type.title()} skin type.")
    
    # Suggest checking the other skin type
    if len(available_skin_types) > 1:
        other_skin = available_skin_types[1 - selected_skin_index]
        st.info(f"💡 Try viewing recommendations for **{other_skin.title()}** skin instead!")
else:
    # Show product count
    st.success(f"Found **{len(recommendations)}** products for {selected_skin_type.title()} skin")
    
    # Display products table
    st.dataframe(
        recommendations.reset_index(drop=True),
        use_container_width=True,
        height=420
    )
    # Statistics
    st.markdown("---")
    st.subheader("Product Statistics")
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.metric("Total Products", len(recommendations))
    with colB:
        avg_price = recommendations['Price_USD'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    with colC:
        avg_rating = recommendations['Rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f} ⭐")
    # Category breakdown
    st.markdown("---")
    st.subheader("📦 Products by Category")
    
    category_counts = recommendations['Category'].value_counts()
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        st.bar_chart(category_counts)
    
    with col_cat2:
        for category, count in category_counts.items():
            st.write(f"**{category}**: {count} products")
    # Card view
    if show_cards:
        st.markdown("---")
        st.subheader("🛍 Product Details")
        
        # Group by category
        for category in recommendations['Category'].unique():
            with st.expander(f"📦 {category} ({len(recommendations[recommendations['Category'] == category])} products)"):
                category_products = recommendations[recommendations['Category'] == category]
                
                for idx, row in category_products.iterrows():
                    col_prod1, col_prod2 = st.columns([2, 1])
                    
                    with col_prod1:
                        st.markdown(f"""
                        ### {row['Product_Name']}
                        🏷️ **Brand**: {row['Brand']}  
                        🧪 **Key Ingredient**: {row['Main_Ingredient']}  
                        🎯 **Suitable for**: {row['Skin_Type'].title()}
                        """)
                    
                    with col_prod2:
                        st.metric("Price", f"${row['Price_USD']}")
                        st.metric("Rating", f"{row['Rating']} ⭐")
                        st.caption(f"{row['Number_of_Reviews']} reviews")
                    
                    st.markdown("---")
# --------------------------------------------------
# QUICK SWITCH BUTTON
# --------------------------------------------------
if len(available_skin_types) > 1:
    st.markdown("---")
    
    other_index = 1 - selected_skin_index
    other_skin = available_skin_types[other_index]
    
    col_switch1, col_switch2, col_switch3 = st.columns([1, 2, 1])
    
    with col_switch2:
        if st.button(
            f"🔄 Switch to {other_skin.title()} Recommendations",
            use_container_width=True,
            type="primary"
        ):
            st.info(f"Please select **{skin_type_labels[other_index]}** above to view those recommendations")
            st.rerun()
            

# --------------------------------------------------
# ADDITIONAL INSIGHTS
# --------------------------------------------------
if ai_skin and ai_skin != USER_SKIN_TYPE:
    with st.expander("💡 Why these recommendations?"):
        st.markdown(f"""
        Our AI detected **{ai_skin.title()} skin** characteristics, while you selected **{USER_SKIN_TYPE.title()}**.
        
        This could mean:
        - You have **combination skin** with traits of both types
        - Your skin condition varies by zone (T-zone vs cheeks)
        - Seasonal or environmental factors affect your skin
        
        **Our hybrid recommendations** include products suitable for both conditions to give you more options!
        """)