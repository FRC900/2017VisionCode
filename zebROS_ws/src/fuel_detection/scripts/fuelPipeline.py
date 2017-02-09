#!/usr/bin/env python

# SciStacvk 
import numpy as np
import matplotlib.pyplot as plt
import cv2

import csv

# Intiialize important info
image = cv2.resize(cv2.imread("test.png"), None, fx=0.45, fy=0.455, interpolation=cv2.INTER_AREA)
hl, sl, vl, hu, su, vu, area_limit = (0, 0, 0, 0, 0, 0, 0)
dp, min_dist, param1, param2, min_radius, max_radius = (0, 0, 0, 0, 0, 0)
file_name = "storePos.csv"

# Read stored positions from CSV
try:
	with open(file_name, 'rt') as fread:
		reader = list(csv.reader(fread))
		hsv = tuple(reader[0])
		hough = tuple(reader[1])
		hl, sl, vl, hu, su, vu, area_limit = [int(i) for i in hsv]
		dp, min_dist, param1, param2, min_radius, max_radius = [int(i) for i in hough]
except:
	print "Cannot Find: {}".format(file_name)
# Callback for trackbar
def nothing(x):
	pass

# Create HSV filter Window
cv2.namedWindow('HSV')
cv2.createTrackbar('HL', 'HSV', hl, 180, nothing)
cv2.createTrackbar('SL', 'HSV', sl, 255, nothing)
cv2.createTrackbar('VL', 'HSV', vl, 255, nothing)
cv2.createTrackbar('HU', 'HSV', hu, 180, nothing)
cv2.createTrackbar('SU', 'HSV', su, 255, nothing)
cv2.createTrackbar('VU', 'HSV', vu, 255, nothing)
cv2.createTrackbar('AREA_LIMIT', 'HSV', area_limit, 1000, nothing)

# Create Hough tracker Window
cv2.namedWindow('HOUGH')
cv2.createTrackbar('DP', 'HOUGH', dp, 20, nothing)
cv2.createTrackbar('MIN_DIST', 'HOUGH', min_dist, 100, nothing)
cv2.createTrackbar('PARAM1', 'HOUGH', param1, 700, nothing)
cv2.createTrackbar('PARAM2', 'HOUGH', param2, 200, nothing)
cv2.createTrackbar('MIN_RADIUS', 'HOUGH', min_radius, 100, nothing)
cv2.createTrackbar('MAX_RADIUS', 'HOUGH', max_radius, 100, nothing)

while(True):
	# Initialize important variables
	hl         = cv2.getTrackbarPos('HL', 'HSV')
	sl         = cv2.getTrackbarPos('SL', 'HSV')
	vl         = cv2.getTrackbarPos('VL', 'HSV')
	hu         = cv2.getTrackbarPos('HU', 'HSV')
	su         = cv2.getTrackbarPos('SU', 'HSV')
	vu         = cv2.getTrackbarPos('VU', 'HSV')
	area_limit = cv2.getTrackbarPos('AREA_LIMIT', 'HSV')
	dp         = cv2.getTrackbarPos('DP', 'HOUGH')
	min_dist   = cv2.getTrackbarPos('MIN_DIST', 'HOUGH')
	param1     = cv2.getTrackbarPos('PARAM1', 'HOUGH')
	param2     = cv2.getTrackbarPos('PARAM2', 'HOUGH')
	min_radius = cv2.getTrackbarPos('MIN_RADIUS', 'HOUGH')
	max_radius = cv2.getTrackbarPos('MAX_RADIUS', 'HOUGH')

	# Make copy of images for HSV and create image mask using HSV values from slider
	hsv   = image.copy()
	mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hu, su, vu]))

	# Perform morphology "open" which performs dilate and expand with a kernal size of 3
	k_size = 3
	kernel = np.ones((k_size,k_size),np.uint8)
	mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	# Find Contours on mask and process contours
	contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
	approx_contours = []
	for c in contours:
		# If contour area is too small skip it
		area = cv2.contourArea(c)
		if area < area_limit: continue

		# Circles and blobs of circles should beat this condition
		epsilon = 0.005*cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,epsilon,True)
		if len(approx) < 20: continue

		# appedn the convex hull of the contours to fix ignoring the dimples
		approx_contours.append(cv2.convexHull(c))

	# Generate contour based mask
	mask2 = np.zeros(mask.shape, np.uint8)
	cv2.drawContours(mask2, approx_contours, -1, (255, 255, 255), -1)
	# cv2.imshow('CONTOUR_MASK', mask2)

	# Mask original image and create a grayscale image for hough circles
	hough = cv2.bitwise_and(hsv, hsv, mask=mask2) # flip to mask if necessary ()
	gray = cv2.cvtColor(cv2.cvtColor(hough, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

	# Find hough circles 
	# dp=3, minDist=20, param1=500, param2=50, minRadius=5, maxRadius=20
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, 
							   minDist=min_dist, param1=param1, param2=param2, 
							   minRadius=min_radius, maxRadius=max_radius)

	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(hough, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(hough, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

	cv2.drawContours(hsv, approx_contours, -1, (100, 255, 100), 2)
	cv2.imshow('HSV', hsv)
	cv2.imshow('HOUGH', hough)

	k = cv2.waitKey(10)
	if k == 27:
		if circles is not None:
			print "Number of Circles: {}".format(len(circles))
		else:
			print "No Circles Found"
		try:
			with open(file_name, 'wt') as fwrite:
				writer = csv.writer(fwrite)
				writer.writerow((hl, sl, vl, hu, su, vu, area_limit))
				writer.writerow((dp, min_dist, param1, param2, min_radius, max_radius))
		except:
			print "Cannot Find: {}".format(file_name)
		break