import cv2 as cv

face=cv.CascadeClassifier("Open_CV (copy)/Projects/haar_face.xml")
c=cv.VideoCapture(0)
while True:
    isTrue,frame=c.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces_rect=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(110,255,10),thickness=1)
    cv.imshow("detected facews",frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
c.release()
cv.destroyAllWindows()