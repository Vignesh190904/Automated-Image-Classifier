import os
import cv2
import numpy as np
import joblib
import pywt
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_cropped_face_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
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

def extract_face_features(image_path):
    cropped_face = get_cropped_face_if_2_eyes(image_path)
    if cropped_face is None:
        return None
    
    cropped_face = cv2.resize(cropped_face, (32, 32))
    img_har = wavelet_transform(cropped_face, 'db1', 5)
    img_har = cv2.resize(img_har, (32, 32))

    combined_img = np.vstack((cropped_face.reshape(32*32*3, 1), img_har.reshape(32*32, 1)))
    return combined_img.flatten()

def extract_features(cleaned_dir):
    features = []
    labels = []
    class_dict = {}
    current_label = 0
    
    for class_name in os.listdir(cleaned_dir):
        class_path = os.path.join(cleaned_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        class_dict[class_name] = current_label
        for image_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            image_path = os.path.join(class_path, image_file)
            feature = extract_face_features(image_path)
            if feature is not None:
                features.append(feature)
                labels.append(current_label)
        current_label += 1

    features = np.array(features)
    labels = np.array(labels)
    os.makedirs('output', exist_ok=True) 
    joblib.dump((features, labels), 'output/features_labels.pkl')
    joblib.dump(class_dict, 'output/class_dictionary.pkl')

if __name__ == '__main__':
    cleaned_dir = 'data/cleaned'
    extract_features(cleaned_dir)
