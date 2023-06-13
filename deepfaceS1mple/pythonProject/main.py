import deepface.DeepFace
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1_path = "img/macron/1.jpeg"
img2_path = "img/poutine/2.jpeg"

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]

# Detect one face
#face = DeepFace.detectFace('img/macron/1.jpeg', target_size=(244, 244), detector_backend='opencv')

# Compare the detection
'''
fig, axs = plt.subplots(3, 2, figsize=(15,10))
axs = axs.flatten()
for i, b in enumerate(backends):
    try:
        face = DeepFace.extract_faces('img/macron/1.jpeg', target_size=(244, 244), detector_backend='b')
        axs[i].imshow(face)
        axs[i].set_title(b)
        axs[i].axis('off')
    except:
        pass
    plt.show()
    '''

# Face Verfications with different models to compare
''' models = ["VGG-Face", "Facenet","Facenet512", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "Ensemble", "SFace"]
deepface = DeepFace.verify(img1_path, img2_path, model_name=models[1])
'''

# face attributes analysis
result = DeepFace.analyze(img1_path)
print(result)