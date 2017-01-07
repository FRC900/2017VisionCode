# edge detection and colorspaces, includes laplacian and sobel filters that are tuned to the pink whiffle ball

import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame1')
kernel = np.ones((5,5),np.uint8)

# create trackbars for color change, tuned to pink whiffle ball
# lower
cv2.createTrackbar('HLo','frame1',120,179,nothing)
cv2.createTrackbar('SLo','frame1',72,255,nothing)
cv2.createTrackbar('VLo','frame1',120,255,nothing)
# upper
cv2.createTrackbar('HUp','frame1',179,179,nothing)
cv2.createTrackbar('SUp','frame1',255,255,nothing)
cv2.createTrackbar('VUp','frame1',255,255,nothing)

while(1):

	# Take each frame
	_, frame = cap.read()

	# get current positions of four trackbars
	hLo = cv2.getTrackbarPos('HLo','frame1')
	sLo = cv2.getTrackbarPos('SLo','frame1')
	vLo = cv2.getTrackbarPos('VLo','frame1')
	hUp = cv2.getTrackbarPos('HUp','frame1')
	sUp = cv2.getTrackbarPos('SUp','frame1')
	vUp = cv2.getTrackbarPos('VUp','frame1')

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# define range of color in HSV
	lower = np.array([hLo,sLo,vLo])
	upper = np.array([hUp,sUp,vUp])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(frame,frame, mask= mask)

	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	laplacian = cv2.Laplacian(closing,cv2.CV_64F)
	sobelx = cv2.Sobel(closing,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(closing,cv2.CV_64F,0,1,ksize=5)

	cv2.imshow('frame',frame)
	cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	cv2.imshow('opening',opening)
	cv2.imshow('closing',closing)
	cv2.imshow('laplacian',laplacian)
	cv2.imshow('sobelx',sobelx)
	cv2.imshow('sobely',sobely)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
