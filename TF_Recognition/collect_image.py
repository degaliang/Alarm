import cv2 as cv
import uuid
import os
import time

labels = ["Alex"]
number_imgs = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_PATH = "images"

for label in labels:
    cap = cv.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    label_dir_path = os.path.join(BASE_DIR, IMAGES_PATH, label)
    if not os.path.exists(label_dir_path):
        os.mkdir(label_dir_path)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(label_dir_path, label + '_' + '{}.jpg'.format(imgnum))
        cv.imwrite(imgname, frame)
        cv.imshow('frame', frame)
        time.sleep(2)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv.destroyAllWindows()