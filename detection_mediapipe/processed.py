import cv2
import mediapipe as mp

# Initialize media pipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Capture video from default camera
cap = cv2.VideoCapture(0)

def process_face(face_img):
    # Process the extracted face here
    # Example: Convert face to grayscale
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return gray_face

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and get the face detection results
    results = face_detection.process(rgb_frame)

    # Extract the face from the frame and process it
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Extract the face
            face_img = frame[y:y+h, x:x+w]
            processed_face = process_face(face_img)
            
            # Replace the original face region with the processed face
            frame[y:y+h, x:x+w] = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)

    # Display the frame with processed face
    cv2.imshow('Processed Face', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
