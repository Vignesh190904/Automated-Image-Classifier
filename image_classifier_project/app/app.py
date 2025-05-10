from flask import Flask, render_template, request, jsonify
import joblib
import cv2
import numpy as np
import pywt
import os

app = Flask(__name__)

# Load trained model and class dictionary
model = joblib.load('output/final_model.pkl')
class_dict = joblib.load('output/class_dictionary.pkl')
inv_class_dict = {v: k for k, v in class_dict.items()}

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_face_if_2_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    # Extract features from the image
    features = extract_features(img)
    if features is None:
        return jsonify({'error': 'No face with 2 eyes detected'}), 400

    # Make prediction
    prediction = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    predicted_class = inv_class_dict[prediction]

    # Prepare the response
    result = {
        'predicted_class': predicted_class,
        'class_probabilities': {inv_class_dict[i]: prob for i, prob in enumerate(proba)}
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
