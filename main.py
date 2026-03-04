import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv.VideoCapture(0)

# Canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0  # store previous finger positions

# Function to calculate distance between two points
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Index finger tip (landmark 8)
        x = int(hand_landmarks.landmark[8].x * frame.shape[1])
        y = int(hand_landmarks.landmark[8].y * frame.shape[0])

        # Thumb tip (landmark 4)
        tx = int(hand_landmarks.landmark[4].x * frame.shape[1])
        ty = int(hand_landmarks.landmark[4].y * frame.shape[0])

        # Draw a dot on fingertip
        cv.circle(frame, (x, y), 5, (255, 255, 0), -1)

        # Check pinch distance to control drawing
        if distance((x, y), (tx, ty)) < 40:  # pinch detected
            if prev_x != 0 and prev_y != 0:
                # Draw neon line
                cv.line(canvas, (prev_x, prev_y), (x, y), (255, 255, 0), 8)

            prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Create neon glow effect
    glow = cv.GaussianBlur(canvas, (25, 25), 15)
    combined = cv.addWeighted(frame, 1, glow, 0.8, 0)
    combined = cv.addWeighted(combined, 1, canvas, 1.2, 0)

    cv.imshow('Neon Hand Drawing', combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
