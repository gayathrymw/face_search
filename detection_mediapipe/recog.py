import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import os
import dlib


# Initialize media pipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

dataset_path = "/home/gayathry/Documents/GitHub/face_search/FACE_REC/new_db/"

known_face_encodings = []
known_face_names = []

# Process each file in the dataset folder
for filename in os.listdir(dataset_path):
    # Ensure the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Construct full file path
        file_path = os.path.join(dataset_path, filename)

        # Load image using face_recognition
        image = face_recognition.load_image_file(file_path)

        # Compute the face encodings for the image
        # This assumes each image has one and only one face.
        # For images with more than one face, you'd need to modify the code.
        face_encodings = face_recognition.face_encodings(image)

        # Ensure at least one face was found
        if face_encodings:
            known_face_encodings.append(face_encodings[0])
            # Extract name from filename (remove file extension)
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

    print("Processed", len(known_face_encodings), "images")

# Capture video from default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and get the face detection results
    results = face_detection.process(rgb_frame)

    # Draw the face detection results on the frame and extract faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            face_img = frame[y:y+h, x:x+w]
            
            face_landmarks = face_recognition.face_landmarks(face_img, model="large")

            if face_landmarks:
            # Face landmarks for the first face in the image
                face_landmark = face_landmarks[0]
                
                # Convert face landmarks to dlib's full_object_detection format
                dlib_landmarks = [dlib.point(p[0], p[1]) for p in face_landmark["chin"]]
                for facial_feature in ["left_eyebrow", "right_eyebrow", "nose_bridge", "nose_tip", "left_eye", "right_eye", "top_lip", "bottom_lip"]:
                    dlib_landmarks.extend([dlib.point(p[0], p[1]) for p in face_landmark[facial_feature]])
                dlib_rect = dlib.rectangle(x, y, x+w, y+h)
                dlib_face_landmark = dlib.full_object_detection(dlib_rect, dlib_landmarks)
                
                # Get face encodings using the landmarks
                face_encoding = face_recognition.face_encodings(face_img, [dlib_face_landmark])[0]
                
            if face_encodings:
                face_encoding = face_encodings[0]
                # Compare the extracted face with our database using cosine similarity
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                # Put recognition name on the frame
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
