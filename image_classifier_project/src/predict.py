import cv2
import numpy as np
import joblib
import pywt
import sys
import os

# Load trained model and class dictionary
model = joblib.load('output/final_model.pkl')
class_dict = joblib.load('output/class_dictionary.pkl')
inv_class_dict = {v: k for k, v in class_dict.items()}

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Replace your get_cropped_face_if_2_eyes with:
def get_cropped_face_if_valid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        nose = nose_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2 and (len(nose) >= 1 or len(mouth) >= 1):
            return roi_color
    return None

def wavelet_transform(img, mode='haar', level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255.0
    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_reconstructed = pywt.waverec2(coeffs_H, mode)
    img_reconstructed *= 255
    img_reconstructed = np.uint8(np.clip(img_reconstructed, 0, 255))
    return img_reconstructed

def extract_features(img):
    cropped_face = get_cropped_face_if_2_eyes(img)
    if cropped_face is None:
        return None
    cropped_face = cv2.resize(cropped_face, (32, 32))
    img_har = wavelet_transform(cropped_face, 'db1', 5)
    img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((cropped_face.reshape(32*32*3, 1), img_har.reshape(32*32, 1)))
    return combined_img.flatten()

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    features = extract_features(img)
    if features is None:
        print("❌ No face with 2 eyes detected.")
        return

    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    predicted_class = inv_class_dict[prediction]

    print(f"\n✅ Predicted Class: {predicted_class}")
    print("\n📊 Class Probabilities:")
    for cls_idx, prob in enumerate(proba):
        print(f"{inv_class_dict[cls_idx]}: {prob:.4f}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    predict_image(image_path)