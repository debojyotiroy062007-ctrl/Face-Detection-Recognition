# Face Detection and Recognition (Task 5)

A Computer Vision project developed using **OpenCV** to perform face detection and identification on static images using Haar Feature-based Cascade Classifiers and Local Binary Patterns Histograms (LBPH).

## Features
- **Face Detection:** Leverages OpenCV's pre-trained Haar Cascade (haarcascade_frontalface_default.xml) to localize frontal faces.
- **Face Recognition:** Uses cv2.face.LBPHFaceRecognizer trained on reference images to predict identity along with confidence scores.
- **Visual Feedback:** Annotates detected faces with green bounding boxes and predicted names directly on the output image.

## Project Structure
\\\	ext
├── faces/                   # Training images grouped by person name
├── face_recognition.py      # Main pipeline script (train + test)
├── haarcascade_frontalface_default.xml # Cascade model file
├── test_image.jpeg          # Target image for recognition
├── output_result.jpg        # Output with bounding boxes and labels
└── .gitignore
\\\

## Requirements
- Python 3.x
- \opencv-python\
- \opencv-contrib-python\
- \
umpy\

Install dependencies:
\\\ash
pip install opencv-python opencv-contrib-python numpy
\\\

## Usage
1. Place reference face photos in \aces/<PersonName>/\.
2. Add the image you want to evaluate as \	est_image.jpeg\ (or \	est_image.jpg\).
3. Run the recognizer:
\\\ash
py face_recognition.py
\\\
4. View the annotated result saved to \output_result.jpg\.
