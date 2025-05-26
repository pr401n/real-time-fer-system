# 😃 Facial Emotion Detection via Webcam

A real-time multi-face emotion detection system using live webcam footage. This project uses a custom-trained deep learning model to recognize facial expressions of multiple people simultaneously.

---

## 📌 Project Summary

This project performs **real-time facial emotion detection** using live webcam input. It is powered by:

- A **custom-trained deep learning model** based on **VGG19**, trained on the **Kaggle Facial Expression Recognition (FER-2013) dataset**
- **OpenCV** for real-time video capture and face detection
- **TensorFlow/Keras** for deep learning and emotion classification

It detects and classifies **seven human emotions** from multiple faces at once, displaying results live on the video stream.

---

## 🧠 Emotions Detected

- 😠 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😄 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral

---

## 📁 Dataset

- **Facial Expression Recognition (FER-2013)**  
  - Source: [Kaggle FER Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
  - Contains 48x48 grayscale images categorized into 7 emotion classes.

---

## 🛠️ Tech Stack

| Feature                | Tool/Library     |
|------------------------|------------------|
| Model Architecture     | VGG19 (custom-trained) |
| Deep Learning Framework| TensorFlow, Keras |
| Webcam & Face Detection| OpenCV           |
| Numerical Processing   | NumPy, Pandas    |

---

## 🚀 How It Works

1. **Training Phase**  
   A VGG19-based convolutional neural network was trained on the FER-2013 dataset, achieving good performance in classifying facial expressions into 7 emotion classes.

2. **Detection Phase**  
   - OpenCV captures video frames from the webcam.
   - Faces are detected using Haar cascades.
   - Each face is cropped and preprocessed.
   - The trained model predicts the emotion of each detected face.
   - The predicted label is displayed above each face in real-time.

---

