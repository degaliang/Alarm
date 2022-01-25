import os
import pickle
import cv2 as cv
import numpy as np
from PIL import Image

cascade_path = 'D:\My_CS_Projects\Alarm\cascades\data\haarcascade_frontalface_alt2.xml'
face_cascade = cv.CascadeClassifier(cascade_path)

# Find the path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Specify the path to the directory that stores all training images
image_dir = os.path.join(BASE_DIR, 'test')

current_id = 0
ids = {}
training_data = [] # Store pixel array of each training image
data_ids = [] # Store id corresponding to each of the image

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg') or file.endswith('JPG'):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            # Create an id for each label
            if label not in ids:
                ids[label] = current_id
                current_id += 1
            label_id = ids[label]

            pil_image = Image.open(path).convert('L') # Load the image and turn it into gray scale
            # Resize imgae before processing to increase accuracy
            size = (550, 550)
            resized_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(resized_image, 'uint8') # Turn the image into an array of pixel values
            
            # Detect faces in the images used to train
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                training_data.append(roi)
                data_ids.append(label_id)
                
with open('labels.pickle', 'wb') as file:
    pickle.dump(ids, file)

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.train(training_data, np.array(data_ids))
recognizer.save("trained.yml")

print(ids)
print("Training finished. Data is stored in file 'trained.yml'")