import pickle
import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('D:\My_CS_Projects\Alarm\cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv.CascadeClassifier('D:\My_CS_Projects\Alarm\cascades\data\haarcascade_eye.xml')
# profile_face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_profileface.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trained.yml')

with open('labels.pickle', 'rb') as file:
    ids = pickle.load(file)

labels = {v:k for k, v in ids.items()}

cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Put the frame in grey scale so that cascade can be used
    # to identify face(specified by the documentation)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Get a list of detected faces
    # The detectMultiScale() method returns a list of rectangles 
    # of all the detected objects (faces in our first case). Each 
    # element in the list represents a unique face. This list contains 
    # tuples, (x, y, w, h), where the x, y values represent the top-left 
    # coordinates of the rectangle, while the w, h values represent the 
    # width and height of the rectangle, respectively.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    # profile_faces = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        img = "my_pic.png"
        cv.imwrite(img, roi_gray)

        # Recognize the face detected with pre-trained model
        label_id, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            face_recognized = labels[label_id]
        else:
            face_recognized = "Unkown person"
        # print(face_recognized)
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2
        cv.putText(frame, face_recognized, (x,y), font, fontScale, color, thickness)

        # Draw a rectangle around the face identified
        color = (0, 0, 255)
        thickness = 2
        start_point = (x, y)
        end_point = (x+w, y+h)
        cv.rectangle(frame, start_point, end_point, color, thickness)

    for (x, y, w, h) in eyes:
        color = (0, 255, 0)
        thickness = 2
        start_point = (x, y)
        end_point = (x+w, y+h)
        cv.rectangle(frame, start_point, end_point, color, thickness)
    
    # for (x, y, w, h) in profile_faces:
    #     color = (255, 0, 0)
    #     thickness = 2
    #     start_point = (x, y)
    #     end_point = (x+w, y+h)
    #     cv.rectangle(frame, start_point, end_point, color, thickness)

    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()