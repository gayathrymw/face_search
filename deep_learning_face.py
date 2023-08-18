import numpy as np
import cv2
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load the VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    return features.flatten()

# Extract features for all images in the dataset and store
dataset_directory = r"data/faces"
extensions = ['.jpg', '.jpeg', '.png']
image_paths = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) 
               if any(f.endswith(ext) for ext in extensions)]
features = [extract_features(img_path) for img_path in image_paths]


def find_similar_images(query_img_path, features, image_paths):
    query_features = extract_features(query_img_path)
    
    similarities = cosine_similarity([query_features], features)
    sorted_indexes = np.argsort(similarities[0])[::-1]
    
    # Get the top similar image
    most_similar_img = image_paths[sorted_indexes[0]]
    return most_similar_img

query_img_path = "data/val/ben_afflek/httpabsolumentgratuitfreefrimagesbenaffleckjpg.jpg"
query_image = cv2.imread(query_img_path)
cv2.imshow('Query Image', query_image)
similar_img_path = find_similar_images(query_img_path, features, image_paths)

# Display the similar image using OpenCV
img = cv2.imread(similar_img_path)
cv2.imshow('Similar Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
