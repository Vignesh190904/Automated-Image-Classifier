import cv2
import os
import shutil

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def is_valid_image(img_path):
    """
    Check if the image contains at least one face and two eyes for validity.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:  # If there are at least 2 eyes detected, it's valid
            return True
    return False

def clean_images(input_dir, output_dir):
    """
    Process all images in the input directory, check if they're valid,
    and copy them to the output directory.
    """
    # Loop over the label directories in the input folder
    for label in os.listdir(input_dir):
        src_folder = os.path.join(input_dir, label)
        dst_folder = os.path.join(output_dir, label)
        
        # Create the label directory in the output folder if it doesn't exist
        os.makedirs(dst_folder, exist_ok=True)
        
        # Loop over the files in each label folder
        for file in os.listdir(src_folder):
            path = os.path.join(src_folder, file)
            
            # Only process the image if it is valid
            if is_valid_image(path):
                shutil.copy(path, os.path.join(dst_folder, file))  # Copy valid image to the output folder
                print(f"Valid image: {file} copied to {dst_folder}")
            else:
                print(f"Invalid image: {file} skipped")

# Set input and output directory paths
input_dir = 'data/raw'
output_dir = 'data/cleaned'

# Clean the images
clean_images(input_dir, output_dir)