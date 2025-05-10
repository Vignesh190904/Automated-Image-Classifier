import cv2
import os
import numpy as np
import pywt
import joblib

from pathlib import Path
from sklearn.preprocessing import LabelEncoder

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

def wavelet_transform(img, mode='haar', level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255.0
    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_reconstructed = pywt.waverec2(coeffs_H, mode)
    img_reconstructed *= 255
    img_reconstructed = np.uint8(img_reconstructed)
    return img_reconstructed

def extract_face_and_eyes(path):
    img = cv2.imread(str(path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_roi = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
        if len(eyes) >= 2:
            return cv2.resize(face_roi, (32, 32))
    return None

X = []
y = []

DATA_DIR = Path("data/cleaned")
class_names = os.listdir(DATA_DIR)

for label in class_names:
    class_dir = DATA_DIR / label
    for img_path in class_dir.glob("*.jpg"):
        cropped_face = extract_face_and_eyes(img_path)
        if cropped_face is not None:
            scaled_raw = cropped_face / 255.0
            wavelet_img = wavelet_transform(cropped_face, 'db1', 5)
            wavelet_img = cv2.resize(wavelet_img, (32, 32)) / 255.0

            combined = np.concatenate((
                scaled_raw.flatten(), wavelet_img.flatten()
            ))
            X.append(combined)
            y.append(label)

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)

# Save features and labels
os.makedirs("model", exist_ok=True)
joblib.dump({"X": X, "y": y, "label_map": le.classes_}, "model/features_labels.pkl")
print(f"Saved feature matrix with shape {X.shape} and labels to model/features_labels.pkl")
