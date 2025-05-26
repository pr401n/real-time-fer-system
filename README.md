# 😃 Facial Emotion Detection via Webcam

A real-time facial emotion detection system using live webcam footage, powered by deep learning and computer vision techniques.

This project uses a **pretrained VGG19 deep learning model** trained on the **Facial Expression Recognition (FER) dataset**, and leverages **OpenCV** for webcam access and live emotion prediction.

---

## 🎯 Project Description

The system detects and classifies human emotions (like happy, sad, angry, surprised, etc.) from live video captured via webcam. It uses:

- **VGG19** pretrained CNN model fine-tuned on the **Facial Expression Recognition dataset**
- **OpenCV** to capture real-time frames from the webcam
- **TensorFlow** and **Keras** for model definition and prediction

---

## 🧪 Libraries Used

- `TensorFlow`
- `Keras`
- `OpenCV` (`cv2`)
- `NumPy`
- `Pandas`

---

## 📊 Dataset Used

- **Facial Expression Recognition (FER-2013)**  
  A widely used public dataset containing grayscale images of facial expressions classified into seven emotions.

---

## ⚙️ How It Works

1. **Model Training**
   - A VGG19 model is trained on the FER-2013 dataset to classify facial expressions.

2. **Webcam Integration**
   - OpenCV captures frames from the system's webcam.

3. **Real-Time Detection**
   - Faces are detected in the frame.
   - Each face is cropped, preprocessed, and fed into the model.
   - The model predicts the emotion label and displays it on the live feed.

---



