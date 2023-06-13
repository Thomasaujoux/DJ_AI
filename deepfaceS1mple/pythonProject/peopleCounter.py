import cv2



class PersonCounter:
    def __init__(self, video_path):
        self.video_path = video_path

    def count_people(self):
        # Charger le modèle de détection de personnes
        net = cv2.dnn.readNetFromCaffe(
            'MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

        # Ouvrir la vidéo
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0
        total_people_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )

            net.setInput(blob)
            detections = net.forward()

            # Compter le nombre de personnes détectées
            people_count = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5 and detections[0, 0, i, 1] == 15:
                    people_count += 1

            total_people_count += people_count

        # Fermeture de la vidéo
        cap.release()

        return total_people_count


# Usage du code
counter = PersonCounter("img/video/hagla_short.mov")


people_count = counter.count_people()

print("Nombre de personnes présentes dans la vidéo:", people_count)
