#!/usr/bin/env python3
"""CodSoft Task 5: face recognition using Haar cascades and LBPH.

This script assumes:
1. A directory named faces/ exists in the project folder.
2. The faces/ directory contains reference persons as folders named after
   each known identity, like faces/Alice/*.jpg.
3. A test image named test_image.jpg or test_image.jpeg exists in the project folder.

The script trains an LBPH recognizer from the reference images, detects faces
in the test image, predicts each face's identity, draws labeled bounding boxes,
and saves the result as output_result.jpg.
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FACES_DIR = BASE_DIR / "faces"
TEST_IMAGE_PATH = None
OUTPUT_RESULT_PATH = BASE_DIR / "output_result.jpg"

# This is the standard Haar cascade provided by OpenCV.
CASCADE_FILE = 'haarcascade_frontalface_default.xml'


class FaceRecognizerApp:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(str(BASE_DIR / 'haarcascade_frontalface_default.xml'))
        if self.face_cascade.empty():
            raise RuntimeError("Unable to load Haar cascade from local file: haarcascade_frontalface_default.xml")

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            raise RuntimeError("cv2.face.LBPHFaceRecognizer_create is not available. Install opencv-contrib-python.")

        self.name_to_id = {}
        self.id_to_name = {}

    def collect_training_data(self):
        """Collect reference images and face crops from the faces/ directory.

        Accepts either:
          - faces/PersonName/*.jpg  (recommended)
          - faces/*.jpg with file stem format PersonName_1.jpg
        """
        samples = []
        labels = []

        if not FACES_DIR.exists():
            raise FileNotFoundError(f"Training directory not found: {FACES_DIR}")

        person_dirs = sorted([p for p in FACES_DIR.iterdir() if p.is_dir()], key=lambda x: x.name)
        image_files = []

        # Use person folders if available.
        if person_dirs:
            for person_dir in person_dirs:
                person_name = person_dir.name
                if person_name.startswith("."):
                    continue
                if person_name not in self.name_to_id:
                    self.name_to_id[person_name] = len(self.name_to_id)
                    self.id_to_name[self.name_to_id[person_name]] = person_name

                image_files = sorted(
                    [
                        p for p in person_dir.glob("*")
                        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
                    ],
                    key=lambda x: x.name,
                )

                for image_path in image_files:
                    face_crop = self.load_face_crop_from_image(image_path)
                    if face_crop is None:
                        continue
                    samples.append(face_crop)
                    labels.append(self.name_to_id[person_name])

        # Fall back to flat files if no subdirectories are used.
        else:
            image_files = sorted(
                [
                    p for p in FACES_DIR.glob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
                ],
                key=lambda x: x.name,
            )

            for image_path in image_files:
                # Support file names like: Alice_1.jpg, Bob_2.png
                person_name = image_path.stem.split("_")[0]
                if person_name not in self.name_to_id:
                    self.name_to_id[person_name] = len(self.name_to_id)
                    self.id_to_name[self.name_to_id[person_name]] = person_name

                face_crop = self.load_face_crop_from_image(image_path)
                if face_crop is None:
                    continue
                samples.append(face_crop)
                labels.append(self.name_to_id[person_name])

        if not samples:
            raise RuntimeError("No face samples could be read from the faces directory. Add reference images in faces/<PersonName>/.")

        return samples, np.array(labels)

    def load_face_crop_from_image(self, image_path):
        """Detect one face from the supplied image and return a grayscale face crop."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            return None

        # Use the first detected face in the reference image.
        x, y, w, h = faces[0]
        return gray[y:y+h, x:x+w]

    def train(self):
        """Train LBPH recognizer on the reference face images."""
        samples, labels = self.collect_training_data()
        if len(samples) == 0:
            raise RuntimeError("No training face crops were found.")

        # The recognizer demands images of the same size.
        normalized_samples = [cv2.resize(sample, (100, 100)) for sample in samples]
        self.recognizer.train(normalized_samples, labels)

    def recognize_test_image(self):
        """Detect faces in the test image, recognize names, and save labeled output."""
        test_jpg = BASE_DIR / "test_image.jpg"
        test_jpeg = BASE_DIR / "test_image.jpeg"

        if test_jpg.exists():
            TEST_IMAGE_PATH = test_jpg
        elif test_jpeg.exists():
            TEST_IMAGE_PATH = test_jpeg
        else:
            raise FileNotFoundError("Test image not found: expected test_image.jpg or test_image.jpeg")

        test_image = cv2.imread(str(TEST_IMAGE_PATH))
        if test_image is None:
            raise RuntimeError(f"Could not read test image: {TEST_IMAGE_PATH}")

        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            print("No faces detected in test_image.jpg.")
            cv2.imwrite(str(OUTPUT_RESULT_PATH), test_image)
            return

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]
            face_crop = cv2.resize(face_crop, (100, 100))
            label_id, confidence = self.recognizer.predict(face_crop)
            predicted_name = self.id_to_name.get(label_id, "Unknown")

            # Draw the bounding box and predicted label.
            cv2.rectangle(test_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                test_image,
                f"{predicted_name} ({confidence:.1f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Save the image with predictions.
        if not cv2.imwrite(str(OUTPUT_RESULT_PATH), test_image):
            raise RuntimeError(f"Failed to save output image: {OUTPUT_RESULT_PATH}")

        print(f"Saved recognized image to {OUTPUT_RESULT_PATH}")


if __name__ == "__main__":
    app = FaceRecognizerApp()
    app.train()
    app.recognize_test_image()
