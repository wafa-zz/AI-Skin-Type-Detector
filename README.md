### AI Skin Type Detection & Beauty Product Recommendation 

This is an **end-to-end Deep Learning application** that detects a user’s **skin type from a face image** and recommends **personalized beauty products**.  
The project combines **Computer Vision(CNN)** and a **rule-based recommendation System**, deployed using **Streamlit** for real-time interaction.

---

### **✨Features**
**📸**Skin type detection from face images (AI-based)
**✍️**Manual skin type selection option
🧠Deep learning model using EfficientNet (Transfer Learning)
📊Confidence-based prediction with probability scores  
🎯Product recommendations based on skin type 
📋Interactive product table (sortable & scrollable)  
📈Basic data visualization (price & rating charts)
💻Clean and user-friendly Streamlit interface

---

### **🛠️ Tech Stack**
- **Python**
- **TensorFlow/Keras**
- **EfficientNet-B0**
- **Pandas & NumPy**
- **Scikit-learn**
- **Streamlit**
- **Pillow (Image Processing)**
- **Matplotlib**

---

### **🤖 Machine Learning Models Used**

#### 🔹 Skin Type Classification
- Model: **EfficientNet-B0**
- Framework: **TensorFlow/Keras**
- Approach: Transfer Learning + Fine-Tuning
- Classes:
  - acne-prone-skin
  - dry-skin
  - oily-skin
  - healthy
- Input: Face Image
- Output: Skin type + Confidence Score

---

### 📂 Dataset Used 

#### 🔹 Image Dataset
- Face images categorized into 4 skin types
- Used for training and validating
- Includes augmented data for better generalization

#### 🔹 Product Dataset
- Stored in **Cosmetic product details**
- Includes:
  - Product Name
  - Brand
  - Category
  - Skin Type
  - Price
  - Rating
  - Number of Reviews
  - Ingredients
  - Usage details

---

### **⚙️ Data Preprocessing**

#### 🔹 Image Data
- Resized images to **224 × 224**
- Normalized (0-1 scaling)
- Data augmentation (rotation, flipping, Zoom)
- Train–validation split (80:20)

#### 🔹 Product Data
- Missing value handling
- Feature engineering (review score, popularity score)
- Ingredient vectorization using **TF-IDF**
- Price normalization

---

### **How the System Works**

1. User selects:
   - 📸 Upload image (AI prediction)
   - ✍️ Manual skin type selection
3. Image is preprocessed and passed to the CNN model
4. Model predicts skin type with confidence scores
5. System filters relevant products from dataset
6. Product are ranked based on rating
7. Results are displayed using Streamlit

---

### 💻 Streamlit Interface 
- Upload face / skin image
- View detected skin type
- See confidence scores
- Get recommended products
- Explore products via:
  1. Interactive tables
  2. Basic charts (price & rating)

---

#### 💡 Conclusion
- This project demonstrates:
  - End-to-end Deep Learning pipeline
  - Real-world AI application development
  - Integration of ML model + recommendation system
  - Deployment using Streamlit


change this based on my updation, if something we can do more to my project please recomment
