import os
import cv2
import numpy as np
import dlib
import faiss

# Define the threshold for similarity
SIMILARITY_THRESHOLD = 0.5

def extract_embeddings(face_recognizer, aligned_face):
    embedding = face_recognizer.compute_face_descriptor(aligned_face)
    return np.array(embedding)

def recognize_face(image, face_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return faces

def align_face(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape_predictor = dlib.shape_predictor("model/shape_data")
    landmarks = shape_predictor(gray, face)

    aligned_face = dlib.get_face_chip(image, landmarks)
    return aligned_face

def load_embeddings(embeddings_dir):
    embeddings = {}
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            user_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(embeddings_dir, filename))
            embeddings[user_id] = embedding
    return embeddings

def build_faiss_index(embeddings):
    # Create a Faiss index and add reference embeddings to it
    d = len(list(embeddings.values())[0])  # Dimension of the embeddings
    index = faiss.IndexFlatL2(d)  # Use L2 (Euclidean) distance

    embeddings_list = list(embeddings.values())
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    index.add(embeddings_array)

    return index

def search_faiss_index(index, query_embedding):
    # Search for similar embeddings in the Faiss index
    D, I = index.search(np.array([query_embedding]).astype('float32'), index.ntotal)

    return D[0], I[0]

def main():
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/Saugat Malla_Image_42.jpg'

    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')

    embeddings = load_embeddings(embeddings_dir)
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            if user_id in embeddings:
                continue
            
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image, face_detector)

            if not faces:
                continue

            aligned_face = align_face(image, faces[0])
            embedding = extract_embeddings(face_recognizer, aligned_face)

            embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
            np.save(embedding_path, embedding)

            print(f"Embedding saved for {filename}")

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detector)

    if not query_faces:
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_recognizer, aligned_face)

    # Build the Faiss index
    index = build_faiss_index(embeddings)

    # Search for similar faces in the Faiss index
    distances, indices = search_faiss_index(index, query_embedding)

    print("Similar images:")
    for i, idx in enumerate(indices):
        similarity = distances[i]  # Using L2 (Euclidean) distance as similarity
        if similarity <= SIMILARITY_THRESHOLD:
            user_id = os.path.splitext(os.listdir(dataset_dir)[idx])[0]
            print(f"User ID: {user_id}, Similarity: {similarity:.4f}")

            similar_image_path = os.path.join(dataset_dir, user_id + '.jpg')
            similar_image = cv2.imread(similar_image_path)

            print(f"Similar Image Filename: {user_id}.jpg")

            cv2.imshow("Similar Image", similar_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
