# Face Detection and Recognition

## CodSoft Artificial Intelligence Internship – Task 5

This project is a Face Detection and Recognition system developed using Python and OpenCV.

The application detects faces through a webcam and recognizes a registered person using the LBPH (Local Binary Patterns Histograms) face recognition algorithm.

## Features

- Real-time face detection
- Face recognition using LBPH
- Webcam-based recognition
- Registered person recognition
- Unknown face detection
- Confidence score display
- Face dataset collection
- AI model training

## Technologies Used

- Python
- OpenCV
- OpenCV Contrib
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer

## Project Structure

```text
CODSOFT_TASK5/
│
├── dataset/
│
├── trainer/
│   ├── collect_faces.py
│   ├── train_model.py
│   ├── face_recognition.py
│   ├── requirements.txt
│   ├── trainer.yml
│   └── labels.json
│
├── README.md
└── .gitignore