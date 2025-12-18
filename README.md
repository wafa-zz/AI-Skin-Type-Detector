### **AI Skin Type Detection & Beauty Product Recommendation **

This is an **end-to-end Machine Learning application** that detects a user’s **skin type from a face image** and recommends **personalized beauty products**.  
The project combines **Deep Learning**, **Data Preprocessing**, and a **Hybrid Recommendation System**, all deployed using **Streamlit** for live interaction.

---

### **Features ✨**
**->** Skin type detection from face images  
**->** Deep learning model using EfficientNet  
**->** Confidence-based skin type prediction  
**->** Personalized beauty product recommendations  
**->** Hybrid recommender (Skin type + Ingredients + Popularity)  
**->** Interactive and user-friendly Streamlit UI  

---

### **Tech Stack 🛠️**
- **Python**
- **PyTorch**
- **EfficientNet (CNN)**
- **Pandas & NumPy**
- **Scikit-learn**
- **Streamlit**
- **OpenCV**
- **Matplotlib**

---

### **Machine Learning Models Used**

#### 🔹 Skin Type Classification
- Model: **EfficientNet-B0**
- Framework: **PyTorch**
- Classes:
  - acne-prone-skin
  - dry-skin
  - oily-skin
  - healthy
- Input: Face image
- Output: Skin type + confidence score

#### 🔹 Product Recommendation System
- Approach: **Hybrid Recommendation**
- Techniques:
  - Skin type–based filtering
  - Ingredient similarity using **TF-IDF**
  - Popularity scoring using ratings and reviews

---

### **Dataset Used **

#### 🔹 Image Dataset
- Face images categorized into 4 skin types
- Used for training and validating the CNN model

#### 🔹 Product Dataset
- Cosmetic product details
- Includes:
  - Skin type
  - Ingredients
  - Price
  - Rating
  - Number of reviews
  - Brand & category

---

### **Data Preprocessing ⚙️**

#### 🔹 Image Data
- Resized images to **224 × 224**
- Normalized using ImageNet statistics
- Data augmentation (rotation, flipping)
- Train–validation split (80:20)

#### 🔹 Product Data
- Missing value handling
- Feature engineering (review score, popularity score)
- Ingredient vectorization using **TF-IDF**
- Price normalization

---

### **How the System Works**

1. User selects a skin type or uploads a face image
2. Image is preprocessed and passed to the CNN model
3. Skin type is predicted with confidence score
4. Product recommender filters and ranks suitable products
5. Results are displayed instantly using Streamlit

---

### **Streamlit Interface **
- Upload face / skin image
- View detected skin type
- Get top product recommendations
- Interactive tables and charts
- Optional card view for product details

