import cv2 as cv 

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


capture = cv.VideoCapture(0)

while True:

    flag,frame= capture.read()

    if not flag:
        break


    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y), (x+w,y+h),(0,0,255), 3)
    cv.imshow('Test', frame)

    cv.waitKey(1)


capture.release()
cv.destroyAllWindows()