#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2

import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

import math
import time
import sys

import ThreadingUtility


class FuelDetector:
	def __init__(self):
	    self.isTesting = False
	    self.bridge = CvBridge()
	    self.pub_detections = rospy.Publisher("/blobs", String, queue_size=1)
	    self.sub_image = rospy.Subscriber("/zed/rgb/image_rect_color", Image, self.processImage, queue_size=1)
	    
	    if len(sys.argv) == 2:
	        def nothing(x):
	            pass
	    self.isTesting = True
	    self.image = np.zeros((720, 1280, 3), np.uint8)
	    cv2.namedWindow('HSV')
	    cv2.createTrackbar('HL', 'HSV', 0, 180, nothing)
        cv2.createTrackbar('SL', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('VL', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('HU', 'HSV', 0, 180, nothing)
        cv2.createTrackbar('SU', 'HSV', 0, 255, nothing)
        cv2.createTrackbar('VU', 'HSV', 0, 255, nothing)
        self.hl = 0
        self.sl = 0
        self.vl = 0
        self.hu = 0
        self.su = 0
        self.vu = 0
        self.window_thread = RacecarUtilities.StoppableThread(target=self.window_runner)
        self.window_thread.start()
    
    rospy.loginfo("BlobDetector initialized.")
    
    def processImage(self, image_msg):
        im = self.bridge.imgmsg_to_cv2(image_msg)
        
        im = im[:.4*len(im)]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        self.msg = String()
        if not self.isTesting:
            self.find_color(im, cv2.inRange(hsv, np.array([45, 110, 100]), np.array([65, 210, 150])))
        self.pub_blobs.publish(self.msg)
        else:
            self.find_color(im, "testing", cv2.inRange(hsv, np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])))
    
    def find_color(self, passed_im, mask):
        im = passed_im.copy()
        if self.isTesting:
            self.image = im
        contours = cv2.findContours(mask, cv2.cv.CV_RETR_TREE, cv2.cv.CV_CHAIN_APPROX_SIMPLE)[0]
        approx_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                pass # flip to continue
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.00005*perim, True)
            if len(approx) > 20:
			    approx_contours.append(approx)
		if approx_contours:
		    if self.isTesting:
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
				        cv2.drawContours(im, approx_contours, -1, (255, 0, 0), 3)
				        cv2.rectangle(im, (minX, minY), (maxX, maxY), (255,255,0), 3)
		
				  
	        
	    
	


