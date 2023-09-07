import os
import cv2
import numpy as np
from annoy import AnnoyIndex
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.preprocessing import normalize

# Step 1: Generate embeddings for all images in the dataset
def generate_embeddings(dataset_folder, model, device):
    image_paths = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder)]
    embeddings = []

    # Set the model to evaluation mode
    model.eval()

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize if necessary
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_image = transform(image).unsqueeze(0).to(device)

        # Generate the embedding
        with torch.no_grad():
            embedding = model(input_image).squeeze().cpu().numpy()
        
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings)  # Normalize the embeddings

    return embeddings

# Step 2: Build an Annoy index for embeddings
def build_annoy_index(embeddings, num_trees=10):
    num_dimensions = embeddings.shape[1]
    annoy_index = AnnoyIndex(num_dimensions, metric='angular')  # You can also use 'euclidean' metric

    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(num_trees)  # Build the index

    return annoy_index

# Step 3: Query the index to find similar images
def find_similar_images(query_embedding, annoy_index, num_neighbors=5):
    similar_indices = annoy_index.get_nns_by_vector(query_embedding, num_neighbors)
    return similar_indices

if __name__ == "__main__":
    dataset_folder = 'dataset'

    # Load the pre-trained ResNet-50 model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = resnet_model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

    embeddings = generate_embeddings(dataset_folder, resnet_model, resnet_model.device)
    annoy_index = build_annoy_index(embeddings)

    # Replace this with the query image's path
    query_image_path = 'eval/yama buddha_Image_53.jpg'
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    # Preprocess and generate the query image's embedding
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    query_input_image = transform(query_image).unsqueeze(0).to(resnet_model.device)
    query_embedding = resnet_model(query_input_image).squeeze().cpu().numpy()
    query_embedding = normalize(np.array([query_embedding]))

    num_neighbors = 5  # Number of similar images to retrieve
    similar_indices = find_similar_images(query_embedding[0], annoy_index, num_neighbors)

    print("Similar Images:")
    for idx in similar_indices:
        similar_image_path = os.path.join(dataset_folder, os.listdir(dataset_folder)[idx])
        print(similar_image_path)
