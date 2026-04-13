import cv2
import mediapipe as mp
import math
import time

# --- Functions ---
def calculate_angle(a, b, c):
    """Calculate angle at point b (in degrees) given points a, b, c"""
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(ba[0], ba[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ba*mag_bc == 0:
        return 0
    angle = math.degrees(math.acos(dot / (mag_ba*mag_bc)))
    return angle

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # webcam
prev_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect and convert to RGB
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get key points
            l_shldr = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            r_shldr = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]

            # Calculate torso angle
            torso_angle = calculate_angle(l_shldr, l_hip, r_hip)

            # Slouch detection
            if torso_angle < 160:  # tweak threshold if needed
                cv2.putText(frame, "Slouching!", (50,50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 3)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        # Show frame
        cv2.imshow('Posture Monitor', frame)

        # Press ESC to exit
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
