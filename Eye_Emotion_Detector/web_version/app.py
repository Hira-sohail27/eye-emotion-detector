import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indexes for MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    # Get eye points
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    # Compute distances
    # vertical
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    # horizontal
    h_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    if h_dist == 0:
        return 0
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

cap = cv2.VideoCapture(0)

print("Eye Emotion Detector Started - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "No Face"
    ear_avg = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            ear_avg = (left_ear + right_ear) / 2.0

            # Simple Emotion Logic based on Eye Aspect Ratio
            if ear_avg < 0.18:
                emotion = "Sleepy / Eyes Closed"
            elif ear_avg < 0.22:
                emotion = "Tired / Sad"
            elif ear_avg < 0.28:
                emotion = "Neutral"
            elif ear_avg < 0.32:
                emotion = "Happy / Active"
            else:
                emotion = "Surprised / Wide Eyes"

            # Draw eye points
            for idx in LEFT_EYE + RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # UI Text
    cv2.rectangle(frame, (0, 0), (400, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear_avg:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (w-180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Eye Emotion Detector - Hira Sohail', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()