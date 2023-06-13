import cv2
from deepface import DeepFace
import time

class VideoAnalyzer:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.total_face_count = 0
        self.total_frame_count = 0
        self.total_time = 0

        DeepFace.build_model('Emotion')

    def analyze_videos(self, max_frames, interval, frames_per_interval):
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            face_count = 0
            frame_count = 0

            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()

                faces = DeepFace.detectFace(frame, detector_backend='mtcnn')

                for face in faces:
                    emotions = DeepFace.analyze(frame)
                    print(emotions)
                    face_count += 1

                frame_count += 1

                end_time = time.time()
                elapsed_time = end_time - start_time
                self.total_time += elapsed_time

                if frame_count % (interval * cap.get(cv2.CAP_PROP_FPS)) == 0:
                    time.sleep(10)

                if frame_count % frames_per_interval == 0:
                    time.sleep(10)

                if frame_count >= max_frames:
                    print('succeeded')
                    cap.release()
                    cv2.destroyAllWindows()

            self.total_face_count += face_count
            self.total_frame_count += frame_count
            cap.release()

    def display_results(self):
        print("Nombre de visages analysés:", self.total_face_count)
        print("Nombre total de frames analysées:", self.total_frame_count)
        print("Temps total:", self.total_time)
        print("Temps moyen par frame:", self.total_time / self.total_frame_count)


# Usage du code
video_paths = ["img/video/hagla_short.mov", "img/video/autre_video.mov"]
max_frames = 4
interval = 10
frames_per_interval = 3

analyzer = VideoAnalyzer(video_paths)
analyzer.analyze_videos(max_frames, interval, frames_per_interval)
analyzer.display_results()
