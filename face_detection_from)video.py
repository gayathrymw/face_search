import cv2 as cv
import numpy as np

img=cv.imread("Open_CV (copy)/Projects/istockphoto-1350474131-612x612.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

har=cv.CascadeClassifier("Open_CV (copy)/Projects/haar_face.xml")

faces_rect=har.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4,minSize=(30,30))
print(f'{len(faces_rect)} face(s) found!')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(110,255,10),thickness=1)
cv.imshow("detected facews",img)

cv.waitKey(0)
