import cv2
import numpy as np
import serial
import time
import sys

# Fix UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Initialize Serial Communication
try:
    Arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)  # Allow Arduino to reset
except serial.SerialException:
    print("Error: Could not open Serial port. Check the COM port.")
    Arduino = None

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")  # More robust

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Drowsiness tracking variables
sleep, drowsy, active = 0, 0, 0
status = "Awake"
color = (0, 255, 0)  # Green

def compute_ear(eye_rect):
    """Compute Eye Aspect Ratio (EAR) approximation."""
    x, y, w, h = eye_rect
    return h / w  # Height-to-width ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        status = "No Face Detected"
        color = (0, 255, 255)  # Yellow
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Improved eye detection
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))
            ear_values = [compute_ear((ex, ey, ew, eh)) for (ex, ey, ew, eh) in eyes]

            # Ensure at least one valid eye detection
            avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0.15  # Lower fallback EAR

            print(f"Avg EAR: {avg_ear:.2f}")  # Debugging print

            # **Drowsiness Detection Logic**ko
            if avg_ear < 0.18:  # Eyes closed (SLEEPING)
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 8:  # Needs continuous 8 frames
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)  # Red.....................................
                    if Arduino:
                        Arduino.write(b'SLEEPING\n')
                    print("⚠️ SLEEPING DETECTED - Sending Alert ⚠️")

            elif 0.18 <= avg_ear <= 0.24:  # Drowsy state
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:  # Needs continuous 6 frames
                    status = "DROWSY !"
                    color = (0, 0, 255)  # Blue
                    if Arduino:
                        Arduino.write(b'DROWSY\n')
                    print("⚠️ DROWSY DETECTED - Sending Warning ⚠️")

            else:  # Eyes Open (AWAKE)
                sleep = 0
                drowsy = 0
                active += 1
                if active > 6:
                    status = "Awake"
                    color = (0, 255, 0)  # Green
                    if Arduino:
                        Arduino.write(b'AWAKE\n')

    # Display the status on screen
    cv2.putText(frame, status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Show the video feed with status
    cv2.imshow("Drowsiness Detection", frame)

    # Exit when 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
