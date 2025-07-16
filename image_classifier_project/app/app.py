from flask import Flask, render_template, request, jsonify
import joblib
import cv2
import numpy as np
import pywt
import os
import pathlib

app = Flask(__name__)

# Project base path (e.g., /opt/render/project/src/)
BASE_PATH = pathlib.Path(__file__).resolve().parent.parent

# Correct relative paths to model and haarcascade folders
MODEL_PATH = BASE_PATH / 'output' / 'final_model.pkl'
DICT_PATH = BASE_PATH / 'output' / 'class_dictionary.pkl'
HAAR_DIR = BASE_PATH / 'haarcascade'

# Load model and class dictionary
model = joblib.load(MODEL_PATH)
class_dict = joblib.load(DICT_PATH)
inv_class_dict = {v: k for k, v in class_dict.items()}

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(str(HAAR_DIR / 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(str(HAAR_DIR / 'haarcascade_eye.xml'))
profile_cascade = cv2.CascadeClassifier(str(HAAR_DIR / 'haarcascade_profileface.xml'))
nose_cascade = cv2.CascadeClassifier(str(HAAR_DIR / 'haarcascade_mcs_nose.xml'))
mouth_cascade = cv2.CascadeClassifier(str(HAAR_DIR / 'haarcascade_mcs_mouth.xml'))

def get_cropped_face_if_valid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try frontal face first
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        # Fallback to profile face
        faces = profile_cascade.detectMultiScale(gray, 1.3, 5)

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
    cropped_face = get_cropped_face_if_valid(img)
    if cropped_face is None:
        print("❌ No valid face detected with required features.")
        return None
    cropped_face = cv2.resize(cropped_face, (32, 32))
    img_har = wavelet_transform(cropped_face, 'db1', 5)
    img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((cropped_face.reshape(32*32*3, 1), img_har.reshape(32*32, 1)))
    return combined_img.flatten()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'fail', 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'fail', 'message': 'No selected file'}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'fail', 'message': 'Could not read image'}), 400

    features = extract_features(img)
    if features is None:
        return jsonify({'status': 'fail', 'message': "Person/Celebrity can't be classified"}), 200

    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    predicted_class = inv_class_dict[prediction]
    confidence = proba[prediction]

    if confidence < 0.80:
        return jsonify({
            'status': 'fail',
            'message': "The model is not confident enough to classify this person."
        }), 200

    result = {
        'status': 'success',
        'predicted_class': predicted_class,
        'class_probabilities': {inv_class_dict[i]: float(prob) for i, prob in enumerate(proba)}
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
