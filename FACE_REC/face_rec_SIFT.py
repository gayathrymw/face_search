import cv2
import numpy as np
import os

# Function to detect and crop face from an image
def detect_and_crop_face(image, face_cascade):
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    if len(faces) == 0:
        return image
    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier('FACE_REC/haar_face.xml')

# Load the input image
input_image = cv2.imread('FACE_REC/eval/2023-08-21-104257.jpg', cv2.IMREAD_GRAYSCALE)
cropped_input = detect_and_crop_face(input_image, face_cascade)
cv2.imshow('Cropped Input', cropped_input)

# Get a list of dataset image paths from the 'dataset' folder
dataset_folder = 'FACE_REC/dataset/new_db'
dataset_image_paths = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder)]

# Initialize SIFT detector and descriptor
sift = cv2.SIFT_create()

# Find the keypoints and descriptors for the cropped input image
keypoints_input, descriptors_input = sift.detectAndCompute(cropped_input, None)

# Initialize a brute-force matcher
bf = cv2.BFMatcher()

best_match_image = None
best_match_distance = float('inf')

# Iterate through the dataset images
for dataset_image_path in dataset_image_paths:
    dataset_image = cv2.imread(dataset_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Find the keypoints and descriptors for the dataset image
    keypoints_dataset, descriptors_dataset = sift.detectAndCompute(dataset_image, None)
    
    # Match descriptors using BFMatcher
    matches = bf.knnMatch(descriptors_input, descriptors_dataset, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Calculate the distance of good matches
    if len(good_matches) > 0:
        distance = sum([match.distance for match in good_matches]) / len(good_matches)
    else:
        distance = float('inf')
    
    # Update best match if the current image is more similar
    if distance < best_match_distance:
        best_match_distance = distance
        best_match_image = dataset_image

# Display the best match image
best_match_image = cv2.resize(best_match_image, (700, 700))
cv2.imshow('Best Match Image', best_match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
