# draws contours around certain HSV parameters, then draw a rectangle around the object
import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame1')
kernel = np.ones((5,5),np.uint8)

# create trackbars for color change
# lower
cv2.createTrackbar('HLo','frame1',120,179,nothing)
cv2.createTrackbar('SLo','frame1',72,255,nothing)
cv2.createTrackbar('VLo','frame1',120,255,nothing)
# upper
cv2.createTrackbar('HUp','frame1',179,179,nothing)
cv2.createTrackbar('SUp','frame1',255,255,nothing)
cv2.createTrackbar('VUp','frame1',255,255,nothing)

cv2.createTrackbar('areaTrackbar','frame1',10000,50000,nothing)

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

	areaTrackbar = cv2.getTrackbarPos('areaTrackbar','frame1')

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# define range of interested (tuned to pink) color in HSV
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

	# contours!
	contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame,contours,-1,(0,255,0),3)

	if len(contours) > 0:
		# loops through each contour in the giant contour array, finds the min and max x and y, then draws a rectangle
		for x in range(0, len(contours)):
			minX = 10000
			minY = 10000
			maxX = 0
			maxY = 0
			for y in range(0, len(contours[x])):

				if contours[x][y][0][0] < minX:
					minX = contours[x][y][0][0]
				if contours[x][y][0][1] < minY:
					minY = contours[x][y][0][1]
				if contours[x][y][0][0] > maxX:
					maxX = contours[x][y][0][0]
				if contours[x][y][0][1] > maxY:
					maxY = contours[x][y][0][1]

			# if the area is above the area trackbar value, display the rectangle
			area = cv2.contourArea(contours[x])
			if area > areaTrackbar:			
				cv2.rectangle(frame,(minX, minY),(maxX, maxY),(255,255,0),3)
				cx = (minX + maxX)//2
				cy = (minY + maxY)//2
				cv2.circle(frame, (cx, cy), 10, (0, 0, 255))


	cv2.imshow('frame',frame)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
