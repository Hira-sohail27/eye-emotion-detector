const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const emotionText = document.getElementById('emotion');
const confidenceText = document.getElementById('confidence');
const bar = document.getElementById('bar');
const startBtn = document.getElementById('startBtn');
const leftEyeEl = document.getElementById('leftEye');
const rightEyeEl = document.getElementById('rightEye');
const blinkEl = document.getElementById('blink');
const allEmoEl = document.getElementById('allEmotions');

let modelsLoaded = false;

async function loadModels() {
  const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
  ]);
  modelsLoaded = true;
  confidenceText.innerText = 'Models loaded. Click Start Camera';
}

async function startCamera() {
  if (!modelsLoaded) await loadModels();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
    video.srcObject = stream;
  } catch (e) {
    alert('Camera permission needed');
  }
}

video.addEventListener('play', () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(overlay, displaySize);

  setInterval(async () => {
    const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    const resized = faceapi.resizeResults(detections, displaySize);
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0,0, overlay.width, overlay.height);
    faceapi.draw.drawDetections(overlay, resized);
    faceapi.draw.drawFaceLandmarks(overlay, resized);

    if (detections.length > 0) {
      const expr = detections[0].expressions;
      const sorted = Object.entries(expr).sort((a,b)=>b[1]-a[1]);
      const top = sorted[0];
      
      emotionText.innerText = top[0];
      confidenceText.innerText = `${(top[1]*100).toFixed(1)}% confidence`;
      bar.style.width = `${top[1]*100}%`;

      allEmoEl.innerText = sorted.slice(0,3).map(s=>`${s[0]}:${(s[1]*100).toFixed(0)}%`).join(' | ');

      // Eye openness from landmarks
      const landmarks = detections[0].landmarks;
      const leftEye = landmarks.getLeftEye();
      const rightEye = landmarks.getRightEye();
      
      const eyeAspect = (eye) => {
        const h1 = Math.hypot(eye[1].x-eye[5].x, eye[1].y-eye[5].y);
        const h2 = Math.hypot(eye[2].x-eye[4].x, eye[2].y-eye[4].y);
        const w = Math.hypot(eye[0].x-eye[3].x, eye[0].y-eye[3].y);
        return (h1+h2)/(2*w);
      }
      const leftEAR = eyeAspect(leftEye);
      const rightEAR = eyeAspect(rightEye);
      leftEyeEl.innerText = leftEAR.toFixed(2);
      rightEyeEl.innerText = rightEAR.toFixed(2);
      blinkEl.innerText = (leftEAR < 0.2 && rightEAR < 0.2) ? 'Blinking / Eyes Closed' : 'Open';

    }
  }, 100);
});

startBtn.addEventListener('click', startCamera);
loadModels();
