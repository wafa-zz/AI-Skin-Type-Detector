### AI Skin Type Detection & Beauty Product Recommendation 

This is an end-to-end Deep Learning application that detects a user’s skin type from a face image and recommends personalized skincare products.

The project combines Computer Vision (CNN) and a rule-based recommendation system, deployed using Streamlit for real-time interaction.
--
##✨ Features 
📸 Skin type detection from face images (AI-based)
✍️ Manual skin type selection option
🧠 Deep Learning model using EfficientNet (Transfer Learning)
📊 Confidence-based prediction with probability scores
🎯 Product recommendations based on skin type
📋 Interactive product table (sortable & scrollable)
📈 Basic data visualization (price & rating charts)
💻 Clean and user-friendly Streamlit interface
--
##🛠️ Tech Stack
Python
TensorFlow / Keras
EfficientNet-B0
Pandas & NumPy
Scikit-learn
Streamlit
Pillow (Image Processing)
Matplotlib
--
##🤖 Machine Learning Model
🔹 Skin Type Classification
Model: EfficientNet-B0
Framework: TensorFlow / Keras
Approach: Transfer Learning + Fine-Tuning
Classes:
  - acne-prone-skin
  - dry-skin
  - oily-skin
  - healthy
Input: Face Image
Output: Skin Type + Confidence Scores
--
##📂 Dataset Used
🔹 Image Dataset
Face images categorized into 4 skin types
Used for training and validation
Includes augmented data for better generalization

🔹 Product Dataset
Stored in products.csv
Includes:
 - Product Name
 - Brand
 - Category
 - Skin Type
 - Price
 - Rating
 - Number of Reviews
 - Ingredients
 - Usage details
--
##⚙️ Data Preprocessing
🔹 Image Data
Resized to 224 × 224
Normalized (0–1 scaling)
Data augmentation:
  - Rotation
  - Flipping
  - Zoom
Train–validation split (80:20)

🔹 Product Data
Filtered based on skin type
Sorted using product ratings
Basic cleaning and formatting

##🔄 How the System Works
1.User selects:
  - Upload image (AI prediction)
  - Manual skin type selection
2.Image is preprocessed and passed to the CNN model
3.Model predicts skin type with confidence scores
4.System filters relevant products from dataset
5.Products are ranked based on rating
6.Results are displayed using Streamlit
--
##💻 Streamlit Interface
Upload face image 📸
View predicted skin type 🧠
See confidence scores 📊
Get recommended products 🎯
Explore products via:
  - Interactive tables
  - Basic charts (price & rating)
--
##📁 Project Structure
project/
│
├── data/
│   └── products.csv
│
├── models/
│   └── skin_classifier.keras
│
├── streamlit_app/
│   └── app.py
│
├── notebooks/
│   ├── 1_train_model.ipynb
│   ├── 2_test_model.ipynb
│   └── 3_recommender.ipynb
│
├── requirements.txt
--
##💡 Conclusion
  - This project demonstrates:
  - End-to-end Deep Learning pipeline
  - Real-world AI application development
  - Integration of ML model + recommendation system
  - Deployment using Streamlit

