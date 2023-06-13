import cv2
import numpy as np
import pyopenpose as op

# Chemin du fichier de configuration OpenPose
params_path = "path/to/openpose/configs/openpose_default.ini"

# Initialiser l'objet OpenPose
openpose = op.WrapperPython()
openpose.configure({"model_folder": "path/to/openpose/models", "config_file": params_path})
openpose.start()

# Chemin de la vidéo
video_path = "img/video/alley_-_39837 (1080p).mp4c"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Variables pour le comptage
frame_count = 0
people_count = 0

while True:
    # Lecture de la frame
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des poses avec OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    openpose.emplaceAndPop([datum])

    # Récupérer les informations des poses détectées
    poses = datum.poseKeypoints
    if poses is not None:
        # Compter le nombre de personnes détectées dans la frame
        people_count = len(poses)

    frame_count += 1

# Fermeture de la vidéo
cap.release()

# Affichage du nombre de personnes détectées
print("Nombre de personnes détectées:", people_count)
