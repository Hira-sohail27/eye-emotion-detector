import math

class EyeAnalyzer:
    '''
    Simple rule-based Eye Emotion Detector
    Based on Eye Aspect Ratio + Eyebrow position
    For better accuracy you can integrate DeepFace inside this class.
    '''
    def detect_emotion(self, left_ear, right_ear, landmarks, w, h):
        avg_ear = (left_ear + right_ear) / 2

        # Eyebrow to eye distance to detect surprise / sad / angry
        # Landmark 105 = left eyebrow, 334 = right eyebrow (mediapipe)
        try:
            left_brow = landmarks.landmark[105]
            left_eye_top = landmarks.landmark[159]
            brow_eye_dist = abs(left_brow.y - left_eye_top.y) * h
        except:
            brow_eye_dist = 20

        # Simple logic - clean and explainable for assignment
        if avg_ear < 0.18:
            return "Sleepy / Eyes Closed", (100, 100, 255)
        elif avg_ear < 0.22 and brow_eye_dist < 15:
            return "Angry / Focused", (0, 0, 255)
        elif avg_ear > 0.32 and brow_eye_dist > 28:
            return "Surprised / Happy", (0, 255, 0)
        elif brow_eye_dist < 18:
            return "Sad / Tired", (255, 150, 0)
        elif avg_ear > 0.28:
            return "Happy / Neutral", (255, 255, 0)
        else:
            return "Neutral", (255, 255, 255)

    def detect_with_deepface(self, face_img):
        # Optional advanced - if you want 7 emotions
        # from deepface import DeepFace
        # result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        # return result[0]['dominant_emotion']
        pass
