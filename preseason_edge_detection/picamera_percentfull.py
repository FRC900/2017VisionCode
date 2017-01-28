# draws contours around certain HSV parameters, then draw a rectangle around the object
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np


def nothing(x):
    pass

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30

rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

cv2.namedWindow('frame1')
kernel = np.ones((5,5),np.uint8)

# create trackbars for color change
# lower
cv2.createTrackbar('HLo','frame1',19,179,nothing)
cv2.createTrackbar('SLo','frame1',57,255,nothing)
cv2.createTrackbar('VLo','frame1',123,255,nothing)
# upper
cv2.createTrackbar('HUp','frame1',52,179,nothing)
cv2.createTrackbar('SUp','frame1',197,255,nothing)
cv2.createTrackbar('VUp','frame1',255,255,nothing)

cv2.createTrackbar('areaTrackbar','frame1',10000,50000,nothing)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

	# Take each frame
	image = frame.array
	#frame = frame[0:340, 60:610]

	# get current positions of four trackbars
	hLo = cv2.getTrackbarPos('HLo','frame1')
	sLo = cv2.getTrackbarPos('SLo','frame1')
	vLo = cv2.getTrackbarPos('VLo','frame1')
	hUp = cv2.getTrackbarPos('HUp','frame1')
	sUp = cv2.getTrackbarPos('SUp','frame1')
	vUp = cv2.getTrackbarPos('VUp','frame1')

	areaTrackbar = cv2.getTrackbarPos('areaTrackbar','frame1')



	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# define range of interested (tuned to pink) color in HSV
	lower = np.array([hLo,sLo,vLo])
	upper = np.array([hUp,sUp,vUp])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower, upper)

	# Bitwise-AND mask and original image
	res = cv2.bitwise_and(image,image, mask= mask)

	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
	laplacian = cv2.Laplacian(closing,cv2.CV_64F)
	sobelx = cv2.Sobel(closing,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(closing,cv2.CV_64F,0,1,ksize=5)

	cv2.imshow("closing", closing)


	# contours!
	contours = cv2.findContours(closing,cv2.cv.CV_RETR_TREE,cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]

	approx_contours = []

	for c in contours:
		moments = cv2.moments(c)
		center = (int(moments['m10']/moments['m00']), (int (moments['m01']/moments['m00'])))
		area = cv2.contourArea(c)
		if area < areaTrackbar: continue
		perim = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.00005*perim, True)

		if len(approx) > 20:
			approx_contours.append(approx)


		for x in range(0, len(approx_contours)):
			minX = 10000
			minY = 10000
			maxX = 0
			maxY = 0
			cx = (minX + maxX)//2
			cy = (minY + maxY)//2

			for y in range(0, len(approx_contours[x])):
				if approx_contours[x][y][0][0] < minX:
					minX = approx_contours[x][y][0][0]
				if approx_contours[x][y][0][1] < minY:
					minY = approx_contours[x][y][0][1]
				if approx_contours[x][y][0][0] > maxX:
					maxX = approx_contours[x][y][0][0]
				if approx_contours[x][y][0][1] > maxY:
					maxY = approx_contours[x][y][0][1]
			if approx_contours:
				cv2.drawContours(image, approx_contours, -1, (255, 0, 0),3)
				cv2.rectangle(image,(minX, minY),(maxX, maxY),(255,255,0),3)

			percentFull = ((340 - minY) / 340.0) * 100
			print(percentFull)


	rawCapture.truncate(0)

	cv2.imshow('image',image)

	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()
