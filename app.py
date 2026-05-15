from flask import Flask, render_template, Response
import cv2
from fer import FER
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

camera = cv2.VideoCapture(0)

detector = FER(mtcnn=True)

def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        # Detect emotions
        result = detector.detect_emotions(frame)

        mood = "No Face"

        if result:

            emotions = result[0]["emotions"]
            mood = max(emotions, key=emotions.get)

            x, y, w, h = result[0]["box"]

            # Face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Mood text
            cv2.putText(
                frame,
                f"Mood: {mood}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        # Convert frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(debug=True)