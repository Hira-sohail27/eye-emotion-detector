'''
Eye Emotion Detector - Clean Python Version
Author: For Saylani IT Assignment
Tech: OpenCV + MediaPipe + DeepFace (optional)
'''

import cv2
import mediapipe as mp
import math
from eye_detector import EyeAnalyzer

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

analyzer = EyeAnalyzer()

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Eye Emotion Detector Started - Press 'q' to quit")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                def get_ear(indices):
                    pts = [(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in indices]
                    # vertical distances
                    v1 = math.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
                    v2 = math.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
                    h_dist = math.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
                    return (v1+v2)/(2*h_dist) if h_dist else 0

                left_ear = get_ear(left_eye_indices)
                right_ear = get_ear(right_eye_indices)
                avg_ear = (left_ear+right_ear)/2

                emotion, color = analyzer.detect_emotion(left_ear, right_ear, face_landmarks, w, h)

                # Draw
                cv2.putText(image, f"Emotion: {emotion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(image, f"Eye Open: {avg_ear:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(image, f"Press Q to exit", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

                # Draw eye contours
                for idx in left_eye_indices + right_eye_indices:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(image, (x,y), 1, (0,255,255), -1)

        cv2.imshow('Eye Emotion Detector - Hira Sohail', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
