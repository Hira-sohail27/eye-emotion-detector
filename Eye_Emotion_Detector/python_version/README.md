# Eye Emotion Detector - Python Version

Clean code for Saylani IT Assignment

### Setup
pip install -r requirements.txt

### Run
python main.py

### Features
- Real-time webcam eye detection using MediaPipe Face Mesh (478 landmarks)
- Eye Aspect Ratio (EAR) calculation
- Blink detection
- Emotion logic based on eye openness + eyebrow distance:
  - Sleepy / Closed
  - Angry / Focused
  - Surprised / Happy
  - Sad / Tired
  - Neutral

### For better accuracy
Uncomment deepface code in eye_detector.py to get 7 emotions: happy, sad, angry, fear, surprise, disgust, neutral.

Press Q to exit.

Made by Hira Sohail
