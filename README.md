# 🎨 Cartoon Character Face Recognition System  
Using Deep Learning & Web Application

## 📌 Overview
The Cartoon Character Face Recognition System is a deep learning–based application designed to identify cartoon characters from facial images. Unlike real human face recognition, cartoon faces vary significantly in artistic style, proportions, and color patterns. This project uses transfer learning with MobileNetV2 to classify cartoon character faces and displays detailed information about the predicted character through a web-based interface.

---

## 🎯 Objectives
- Build a deep learning model capable of recognizing cartoon character faces  
- Classify images into predefined cartoon character categories  
- Develop a user-friendly web application for image upload and prediction  
- Display detailed information about the recognized character  
- Run and deploy the system locally using a Python virtual environment  

---

## 🧠 Problem Statement
Cartoon characters differ widely in:
- Facial structure  
- Artistic style  
- Expressions and exaggerations  
- Color and texture patterns  

Traditional face recognition methods struggle with such variability. Hence, a deep learning–based approach is required to effectively learn and classify cartoon face features.

---

## 📂 Dataset Description

### 📌 Cartoon Characters Used (13 Classes)
- Doraemon  
- Shinchan  
- Tom  
- Jerry  
- Mickey Mouse  
- Pikachu  
- Minion  
- SpongeBob  
- Bugs Bunny  
- Donald Duck  
- Popeye  
- Scooby-Doo  
- Garfield  

### 📁 Dataset Structure
cartoon_dataset/  
├── train/  
├── val/  
└── test/  

Each folder contains subfolders named after the cartoon characters, holding corresponding face images.

---

## ⚙️ Data Preprocessing
- Image resizing to 224 × 224 pixels  
- RGB color normalization  
- Pixel scaling (0–1 range)  
- Dataset split into training, validation, and testing sets  

---

## 🏗️ Model Architecture
The system uses MobileNetV2, a lightweight and efficient CNN architecture.

### 🔹 Key Features
- Pretrained on ImageNet  
- Depthwise separable convolutions  
- Low computational cost  
- Suitable for CPU-based systems  

### 🔹 Training Strategy
- Transfer learning approach  
- Custom classification layers  
- Softmax activation for multi-class classification  

---

## 🏋️ Model Training
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  
- Training on CPU  
- Validation monitoring during training  
- Final trained model saved as model.h5  

---

## 🌐 Web Application
A Streamlit-based web application allows users to interact with the trained model.

### ✨ Features
- Upload cartoon face image  
- Display uploaded image  
- Predict cartoon character  
- Show confidence score  
- Display detailed character information:
  - Character name  
  - Cartoon series  
  - Creator  
  - First appearance  
  - Personality traits  
  - Special abilities  
  - Fun facts  

---

## 🧩 System Architecture
1. User uploads an image via web interface  
2. Image is preprocessed  
3. Trained model predicts the character  
4. Character information is retrieved from a predefined dictionary  
5. Results are displayed on the web page  

---

## 🛠️ Technologies Used

| Component | Technology |
|--------|-----------|
| Programming Language | Python 3.10 |
| Deep Learning Framework | TensorFlow / Keras |
| Model Architecture | MobileNetV2 |
| Web Framework | Streamlit |
| Image Processing | Pillow, NumPy |
| Environment | Python Virtual Environment (venv) |
| Platform | Windows |

---

## ✅ Advantages
- Fast and accurate recognition  
- Lightweight model suitable for CPU systems  
- Interactive and user-friendly web interface  
- Easily expandable with new characters  

---

## 🎯 Applications
- Educational tools for children  
- Entertainment applications  
- AI & computer vision demonstrations  
- Cartoon-based recommendation systems  

---

## ⚠️ Limitations
- Performance depends on dataset quality  
- Limited to predefined characters  
- Accuracy may reduce for low-quality images  
- Retraining required to add new characters  

---

## 🚀 Future Enhancements
- Add more cartoon characters  
- Improve accuracy using larger datasets  
- Deploy the application online  
- Display top-3 predictions  
- Integrate real-time webcam recognition  
- Improve UI/UX design  

---

## 🏁 Conclusion
This project demonstrates the effective use of deep learning for cartoon face recognition. By combining transfer learning, computer vision, and a web-based interface, the system provides accurate predictions along with informative character details.

---

## 👤 Author
Om Kumbharkar
