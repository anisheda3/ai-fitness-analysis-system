
# ===============================
# main_project.py (FIXED VERSION)
# ===============================

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) -               np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle


class ExerciseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # -------------------------------
    # FIXED SQUAT REP DETECTION LOGIC
    # -------------------------------
    def _detect_rep(self, exercise, angles, stage):
        rep_completed = False

        if exercise == "squat" and "knee" in angles:
            # DOWN position
            if angles["knee"] < 100 and stage == "up":
                stage = "down"

            # UP position → one full rep
            elif angles["knee"] > 160 and stage == "down":
                stage = "up"
                rep_completed = True

        return stage, rep_completed

    def analyze_video_real_time(self, video_path, exercise="squat"):
        cap = cv2.VideoCapture(video_path)
        rep_count = 0
        stage = "up"   # IMPORTANT: must start from 'up'

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            angles = {}

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                hip = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                ]
                knee = [
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ]
                ankle = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
                ]

                knee_angle = calculate_angle(hip, knee, ankle)
                angles["knee"] = knee_angle

                stage, rep_done = self._detect_rep(exercise, angles, stage)
                if rep_done:
                    rep_count += 1

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(
                    image,
                    f"Reps: {rep_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Exercise Analysis", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        return rep_count


# Optional direct run (NOT required for Streamlit)
if __name__ == "__main__":
    analyzer = ExerciseAnalyzer()
    print("Total squats counted:",
          analyzer.analyze_video_real_time("demo_squat.mp4"))
