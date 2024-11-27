import os
import face_recognition
import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from PIL import Image
import shutil  # For moving files

# Enable GPU options and mixed precision
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def preprocess_image(image_path):
    """Preprocess the image for better detection."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def detect_faces_with_mtcnn(image, mtcnn_detector):
    """MTCNN face detection."""
    faces = mtcnn_detector.detect_faces(image)
    face_locations = [
        (face['box'][1], face['box'][0] + face['box'][2],  # top, right
         face['box'][1] + face['box'][3], face['box'][0])  # bottom, left
        for face in faces
    ]
    return face_locations

def process_image_with_mtcnn(image_path):
    """Run MTCNN detection and preprocessing concurrently."""
    mtcnn_detector = MTCNN()
    image = preprocess_image(image_path)
    face_locations = detect_faces_with_mtcnn(image, mtcnn_detector)
    return image, face_locations

def encode_faces(image, face_locations):
    """Extract face encodings from detected locations."""
    return face_recognition.face_encodings(image, face_locations)

def process_multiple_references(reference_folder, folder_path, destination_folder):
    # Ensure destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    mtcnn_detector = MTCNN()

    # Loop through each reference image in the reference folder
    for ref_filename in os.listdir(reference_folder):
        ref_image_path = os.path.join(reference_folder, ref_filename)

        if not (ref_filename.endswith(".jpg") or ref_filename.endswith(".png") or ref_filename.endswith(".jpeg")):
            continue  # Skip non-image files

        print(f"Processing reference image: {ref_filename}")

        try:
            # Process and encode the reference image
            ref_image, ref_face_locations = process_image_with_mtcnn(ref_image_path)
            ref_face_encodings = encode_faces(ref_image, ref_face_locations)

            if len(ref_face_encodings) == 0:
                print(f"No face found in the reference image {ref_filename}. Skipping.")
                continue

            person_encoding = ref_face_encodings[0]

            # Loop through all images in the target folder
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)

                if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
                    continue  # Skip non-image files

                print(f"Comparing {filename} against {ref_filename}...")

                try:
                    group_image, group_face_locations = process_image_with_mtcnn(image_path)
                    group_face_encodings = encode_faces(group_image, group_face_locations)

                    # Compare the reference encoding with all faces in the current image
                    results = face_recognition.compare_faces(group_face_encodings, person_encoding, tolerance=0.5)

                    if any(results):
                        print(f"Match found in {filename} for {ref_filename}. Moving to matched folder.")
                        shutil.move(image_path, os.path.join(destination_folder, filename))
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        except Exception as e:
            print(f"Error processing reference image {ref_filename}: {str(e)}")

# Example usage
reference_folder = "reference_images"   # Folder containing multiple reference images
folder_path = "images_folder"           # Folder containing images to check
destination_folder = "matched_images"  # Folder to store all matched images

process_multiple_references(reference_folder, folder_path, destination_folder)
