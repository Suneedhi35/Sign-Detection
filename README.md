# Sign-Detection
# 🤟 AI-Enabled Indian Sign Language Communication System

An AI-powered real-time Indian Sign Language (ISL) recognition and communication system that uses a webcam to detect hand gestures and convert them into readable text.

The system combines computer vision, deep learning, speech processing, multilingual translation, and text-to-speech technologies to provide an interactive communication platform for ISL users.

---

## 🚀 Overview

Communication barriers can make interaction difficult for people who rely on sign language. This project aims to bridge that gap by recognizing Indian Sign Language gestures in real time and providing multiple ways to communicate the recognized information.

The application provides:

- 🤟 Real-time ISL gesture recognition
- 🔤 Alphabet and number recognition
- 📝 Optional word-level recognition
- 🌐 Multilingual text translation
- 🎤 Speech-to-text conversion
- 🔊 Text-to-speech conversion
- 📚 ISL dictionary
- 👤 User registration and authentication
- 🛡️ Admin user management
- 🔗 Integrated sign → translation → speech pipeline

---

## ✨ Features

### 🎥 Real-Time ISL Detection

The application accesses the webcam and detects hand gestures in real time.

The detection pipeline uses:

**Webcam → YOLOv8 → Hand Crop → Vision Transformer → ISL Prediction**

YOLOv8 identifies the hand region and creates a bounding box around the detected hand. The cropped hand image is then passed to the Vision Transformer classifier.

The current classifier supports:

- Numbers: `1–9`
- Alphabets: `A–Z`

A confidence threshold can be adjusted directly from the application.

---
