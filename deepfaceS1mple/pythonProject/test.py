import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
cap = cv2.VideoCapture(0)

## Setup mediapipe instances
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    # Initialize variables for wrist and leg movement calculation
    prev_left_wrist_y = 0
    prev_right_wrist_y = 0
    prev_left_knee_y = 0
    prev_right_knee_y = 0
    prev_left_arm_y = 0  # Added this line
    prev_right_arm_y = 0  # Added this line
    prev_head_y = 0

    wrist_movement_threshold = 0.5
    leg_movement_threshold = 0.5
    arm_movement_threshold = 0.5
    head_movement_threshold = 0.5
    wrist_movement_intensity = 0
    leg_movement_intensity = 0
    arm_movement_intensity = 0
    head_movement_intensity = 0
    num_frames = 0
    wrist_intensities = []
    leg_intensities = []
    arm_intensities = []
    head_intensities = []

    # Setup DrawingSpec for drawing the skeleton
    draw_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))

    # Définir la taille de la fenêtre glissante (par exemple, 10 trames)
    window_size = 10

    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make pose detection
        pose_results = pose.process(image)

        # Make face detection
        face_results = face_detection.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            if pose_results.pose_landmarks is not None:
                landmarks = pose_results.pose_landmarks.landmark
                # ... (Your existing code to calculate sitting down)

                # Detect wrist movement
                left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                wrist_movement_intensity = abs(left_wrist_y - prev_left_wrist_y) + abs(right_wrist_y - prev_right_wrist_y)

                # Detect leg movement
                left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
                right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
                leg_movement_intensity = abs(left_knee_y - prev_left_knee_y) + abs(right_knee_y - prev_right_knee_y)

                # Detect arm movement
                left_elbow_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                right_elbow_y = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
                arm_movement_intensity = abs(left_elbow_y - prev_left_arm_y) + abs(right_elbow_y - prev_right_arm_y)

                # Detect head movement
                head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
                head_movement_intensity = abs(head_y - prev_head_y)

                # Update previous positions
                prev_left_wrist_y = left_wrist_y
                prev_right_wrist_y = right_wrist_y
                prev_left_knee_y = left_knee_y
                prev_right_knee_y = right_knee_y
                prev_left_arm_y = left_elbow_y
                prev_right_arm_y = right_elbow_y
                prev_head_y = head_y

                # Define thresholds for wrist, leg, arm, and head movement intensity levels
                low_intensity_threshold = 0.050
                high_intensity_threshold = 0.100

                # Accumuler les valeurs d'intensité dans une liste
                wrist_intensities.append(wrist_movement_intensity)
                leg_intensities.append(leg_movement_intensity)
                arm_intensities.append(arm_movement_intensity)
                head_intensities.append(head_movement_intensity)

                # Assurer que la liste ne dépasse pas la taille de la fenêtre glissante
                if len(wrist_intensities) > window_size:
                    wrist_intensities.pop(0)
                    leg_intensities.pop(0)
                    arm_intensities.pop(0)
                    head_intensities.pop(0)

                # Calculer la moyenne sur la fenêtre glissante
                avg_wrist_intensity = sum(wrist_intensities) / len(wrist_intensities)
                avg_leg_intensity = sum(leg_intensities) / len(leg_intensities)
                avg_arm_intensity = sum(arm_intensities) / len(arm_intensities)
                avg_head_intensity = sum(head_intensities) / len(head_intensities)

                # Déterminer les niveaux d'intensité des mouvements
                if avg_wrist_intensity < low_intensity_threshold:
                    wrist_intensity_text = "Not moving wrists much"
                elif avg_wrist_intensity < high_intensity_threshold:
                    wrist_intensity_text = "Moving wrists moderately"
                else:
                    wrist_intensity_text = "Moving wrists a lot"

                if avg_leg_intensity < low_intensity_threshold:
                    leg_intensity_text = "Not moving legs much"
                elif avg_leg_intensity < high_intensity_threshold:
                    leg_intensity_text = "Moving legs moderately"
                else:
                    leg_intensity_text = "Moving legs a lot"

                if avg_arm_intensity < low_intensity_threshold:
                    arm_intensity_text = "Not moving arms much"
                elif avg_arm_intensity < high_intensity_threshold:
                    arm_intensity_text = "Moving arms moderately"
                else:
                    arm_intensity_text = "Moving arms a lot"

                if avg_head_intensity < low_intensity_threshold:
                    head_intensity_text = "Not moving head much"
                elif avg_head_intensity < high_intensity_threshold:
                    head_intensity_text = "Moving head moderately"
                else:
                    head_intensity_text = "Moving head a lot"

                # Afficher les textes d'intensité sur l'image
                cv2.putText(image, wrist_intensity_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, leg_intensity_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, arm_intensity_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, head_intensity_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Déterminer si les gens dansent
                dancing_text = ""
                if avg_leg_intensity >= high_intensity_threshold and avg_arm_intensity >= high_intensity_threshold:
                    dancing_text = "People are dancing!"
                else:
                    dancing_text = "People are not dancing."

                cv2.putText(image, dancing_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Obtenir le nombre de visages détectés dans l'image
            num_faces = 0
            if face_results.detections:
                num_faces = len(face_results.detections)

            cv2.putText(image, f"Number of Faces: {num_faces}", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error: {str(e)}")

        # Dessiner le squelette sur l'image
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            draw_spec,
            draw_spec
        )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
