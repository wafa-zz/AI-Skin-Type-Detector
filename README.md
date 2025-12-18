**AI Skin Type Detection & Beauty Product Recommendation System**
An end-to-end Machine Learning application that detects a user’s skin type from a face image and recommends suitable beauty products using a hybrid recommendation approach. The system is deployed using Streamlit for real-time interaction.


**Project Features**

**->** Skin type classification from face images

**->** Deep Learning model using EfficientNet

**->** Personalized beauty product recommendations

**->** Hybrid recommender (Skin Type + Ingredients + Popularity)

**->** Interactive Streamlit web application


**Machine Learning Models Used**

**1️⃣ Skin Type Classification**

Model: EfficientNet-B0

Framework: PyTorch

Classes:

acne-prone-skin

dry-skin

oily-skin

healthy

Input: Face image

Output: Skin type + confidence scores

**2️⃣ Product Recommendation System**

Approach: Hybrid Recommendation

Techniques Used:

**->**Rule-based skin type filtering

**->**Ingredient similarity using TF-IDF

**->**Product popularity scoring (ratings & reviews)

**🧪 Data Preprocessing**
**🔹 Image Data**

Resizing images to 224 × 224

Normalization using ImageNet statistics

Data augmentation (rotation & flipping)

Train–validation split (80:20)

**🔹 Product Data**

Missing value handling

Feature engineering (review score, popularity score)

Price normalization

Ingredient text vectorization using TF-IDF

**⚙️ How It Works**

User uploads a face image via Streamlit UI

Image is preprocessed and passed to the CNN model

Skin type is predicted using EfficientNet

Product recommender filters and ranks suitable products

Results are displayed in real time

**🖥️ Streamlit Interface**

Upload face image

View predicted skin type

Get personalized beauty product recommendations

Interactive and user-friendly UI

**🛠️ Tech Stack**

Python

PyTorch

EfficientNet

Scikit-learn

Pandas & NumPy

Streamlit

Matplotlib


Ingredient similarity using TF-IDF

Product popularity scoring (ratings & reviews)
