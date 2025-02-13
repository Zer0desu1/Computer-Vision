import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr


img = cv2.imread("t.jpg")
cv2.imshow("Original img", img)

grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray image", grayImg)

bfilter = cv2.bilateralFilter(grayImg, 11,17 ,17)
#cv2.imshow("filtered img ", bfilter)


edgedImg = cv2.Canny(bfilter,30,200)
#cv2.imshow("edged img ", edgedImg)

contours,_ = cv2.findContours(edgedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10 , True)
    if len(approx) == 4:
        location = approx
        break  


print(" location : ", location)

mask = np.zeros(img.shape, np.uint8)
maskedImg = cv2.drawContours(mask, [location],0, (255, 255 ,255), -1)
#cv2.imshow("masked", maskedImg)
maskedImg=cv2.cvtColor(maskedImg,cv2.COLOR_BGR2GRAY)

cropedImg = cv2.bitwise_and(img,img, mask = maskedImg)
cv2.imshow("cropped ", cropedImg)

reader = easyocr.Reader(['en'])
result = reader.readtext(cropedImg)


print(result)
cv2.waitKey(0)
cv2.destroyAllWindows()