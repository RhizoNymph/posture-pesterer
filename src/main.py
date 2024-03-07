import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

bad_posture_start_time = None
bad_posture_duration = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            # Calculate angles
            neck_angle = calculate_angle(left_ear, nose, right_ear)
            shoulder_angle = calculate_angle(left_shoulder, nose, right_shoulder)
            head_tilt_angle = calculate_angle(left_ear, nose, right_ear)
            shoulder_alignment = abs(left_shoulder[1] - right_shoulder[1])

            # Check for bad posture conditions
            head_turn_threshold = 20
            head_tilt_lower_threshold = 150
            head_tilt_upper_threshold = 210
            shoulder_alignment_threshold = 0.05

            bad_posture = False
            if abs(neck_angle - shoulder_angle) > head_turn_threshold:
                bad_posture = True
            elif head_tilt_angle < head_tilt_lower_threshold or head_tilt_angle > head_tilt_upper_threshold:
                bad_posture = True
            elif shoulder_alignment > shoulder_alignment_threshold:
                bad_posture = True

            # Display measurements and calculations
            cv2.putText(image, f"Neck Angle: {neck_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Shoulder Angle: {shoulder_angle:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Head Tilt Angle: {head_tilt_angle:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Shoulder Alignment: {shoulder_alignment:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if bad_posture:
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()

                bad_posture_duration = time.time() - bad_posture_start_time

                if bad_posture_duration > 5:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), -1)
                    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
                    cv2.putText(image, "Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                bad_posture_start_time = None
                bad_posture_duration = 0
                cv2.putText(image, "Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()