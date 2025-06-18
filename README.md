<p align="center">
  <img src="https://raw.githubusercontent.com/afzalt3ch/banner.png/main/Gemini_Generated_Image_hb31fqhb31fqhb31.png" alt="WeeBee Banner" width="100%" />
</p>

# 😊 Emotion Detection – Real-Time Facial Emotion Recognition

This project detects human emotions in real-time from webcam feed using a Convolutional Neural Network (CNN). It uses Haar cascades for face detection and a trained Keras model for classifying 7 basic emotions.

---

## 📷 Demo Screenshot

> Accuracy graph and sample interface

![Model Accuracy](https://github.com/afzalt3ch/emotion-detection/blob/main/imgs/accuracy.png)
![App UI](https://github.com/afzalt3ch/emotion-detection/blob/main/imgs/emotion_demo.png)

---

## ✨ Features

- 🎥 Real-time webcam input for live emotion prediction
- 🧠 CNN-based model trained on FER2013 dataset
- 📊 Emotion classes: `Angry`, `Disgusted`, `Fearful`, `Happy`, `Sad`, `Surprised`, `Neutral`
- 🗂️ FER2013 dataset preprocessing script included
- 🎞️ Emotion-specific demo videos available in `static/videos/`
- 🧪 Accuracy plot included in `screenshots/`
- 🖼️ Haar cascade used for face detection before classification

---

## 🛠️ Tech Stack

- Python, Flask
- Keras (TensorFlow backend)
- OpenCV for face detection
- HTML/CSS (Flask templates)
- FER2013 dataset (preprocessed)

---

## 🧪 How It Works

1. Haar cascade detects faces from the live webcam feed.
2. The face region is resized to `48x48` grayscale.
3. The CNN model (`model.h5`) predicts emotion via softmax output.
4. Emotion with highest probability is displayed on the UI.

---

## 📁 Folder Structure

```
emotion-detection/
├── model.h5                       # Trained Keras model
├── haarcascade_frontalface_default.xml
├── dataset_prepare.py            # CSV to image dataset conversion
├── src/
│   ├── data/train/fer2013.csv
│   ├── static/videos/            # Demo MP4s per emotion
│   └── templates/index.html      # Frontend UI
├── imgs/                         # Accuracy chart
├── app.py                        # Flask backend
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/afzalt3ch/emotion-detection.git
cd emotion-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Visit [http://localhost:5000](http://localhost:5000) and allow webcam access.

---

## 📦 Requirements

```txt
flask
tensorflow
keras
opencv-python
numpy
pandas
```

---

## 📜 License

MIT License

---

<p align="center">Made with ❤️ by <strong>Afzal T3ch</strong></p>
