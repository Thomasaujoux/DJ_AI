import cv2
from deepface import DeepFace
import time

# Charger le modèle DeepFace
DeepFace.build_model('Emotion')

# Chemin de la vidéo
video_path = "img/video/hagla_short.mov"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Nombre maximal de frames à traiter
max_frames = 4
interval = 10  # Intervalles de temps en secondes
frames_per_interval = 3  # Nombre de frames à analyser par intervalle

# Variables pour le comptage
frame_count = 0
face_count = 0

total_time = 0


while frame_count < max_frames:
    # Lecture de la frame
    ret, frame = cap.read()
    if not ret:
        break

    # Mesurer le temps de début
    start_time = time.time()

    # Détection des visages
    faces = DeepFace.detectFace(frame, detector_backend='mtcnn')

    for face in faces:
        # Analyse de l'émotion du visage
        emotions = DeepFace.analyze(frame)
        print(emotions)
        face_count += 1

    frame_count += 1

    # Mesurer le temps écoulé
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_time += elapsed_time
    print("-before1")
    # Vérifier si l'intervalle de temps est atteint
    if frame_count % (interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
        # Faire une pause de 10 secondes
        time.sleep(10)
    print('if1')
    # Vérifier si le nombre de frames par intervalle est atteint
    if frame_count % frames_per_interval == 0:
        # Faire une pause de 10 secondes
        time.sleep(10)
    print('if2')
    if frame_count >= max_frames:
        print('succed')
        cap.release()
        cv2.destroyAllWindows()

# Fermeture de la vidéo et affichage du nombre de visages analysés

print("Nombre de visages analysés:", face_count)
print("Temps total:", total_time)
print("Temps moyen par frame:", total_time / frame_count)