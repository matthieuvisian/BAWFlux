import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #
    # Below is mine
    #
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,127,255,0)
    gray2 = gray.copy()
    contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if 200<cv2.contourArea(cnt)<5000:
            (x,y,w,h) = cv2.boundingRect(cnt)
            cv2.rectangle(gray2,(x,y),(x+w,y+h),0,-1)
    #
    # End of mine
    #

    # define range of white color in HSV
    # change it according to your need !
    lower_white = np.array([0,0,0])
    upper_white = np.array([50,50,100])
#    lower_white = np.array([103,59,24], dtype=np.uint8)
#    upper_white = np.array([92,47,20], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('IMG',gray2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
