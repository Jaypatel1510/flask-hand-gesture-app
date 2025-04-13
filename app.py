import cv2
import mediapipe as mp
import threading
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Global variable to hold the latest frame
frame = None
frame_interval = 1  # Process every 10th frame
frame_counter = 0

# Function to capture webcam frames
def capture_frame():
    global frame, frame_counter
    while True:
        success, f = cap.read()
        if not success:
            continue

        frame_counter += 1
        if frame_counter % frame_interval != 0:
            continue  # Skip this frame

        frame = f

# Function to process frames and apply hand gesture recognition
def process_frame():
    global frame
    while True:
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    # Count fingers up
                    fingers_up = count_fingers(hand)

                    # Example control based on fingers
                    if fingers_up == 5:
                        print("🌀 Gesture: FAN ON")
                    elif fingers_up == 1:
                        print("💡 Gesture: LIGHT ON")
                    elif fingers_up == 3:
                        print("🔊 Gesture: ULTRASONIC SENSOR")
                    elif fingers_up == 0:
                        print("❌ Gesture: TURN OFF ALL")
                    else:
                        print(f"👉 Gesture: {fingers_up} finger(s) up")

# Function to count fingers
def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    count = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
    return count

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Video streaming generator to serve frames to Flask
def gen_frames():
    while True:
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_data = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

# Route to stream video feed
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the capture and processing threads
    capture_thread = threading.Thread(target=capture_frame)
    process_thread = threading.Thread(target=process_frame)

    capture_thread.start()
    process_thread.start()

    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

    # Cleanup
    cap.release()