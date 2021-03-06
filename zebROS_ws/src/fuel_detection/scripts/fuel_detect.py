#!/usr/bin/env python

# SciStacvk 
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ROS imports
import message_filters
import rospy

# Point Cloud creation
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32

# Image Processing
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

# Python Utilities
import math
import time
import sys
from threading import Thread
from time import sleep


class BlobDetector:
    def __init__(self):
        self.isTesting  = False
        self.hl         = 20
        self.sl         = 137
        self.vl         = 80
       	self.hu         = 149
        self.su         = 255
        self.vu         = 255
        self.area_limit = 391

        self.dp         = 2
        self.min_dist   = 9
        self.param1     = 246
        self.param2     = 48
        self.min_radius = 5
        self.max_radius = 27

        if len(sys.argv) == 2:
            def nothing(x):
                pass
            self.isTesting = True
            self.contour_image = np.zeros((720, 1280, 3), np.uint8)
            self.hough_image   = np.zeros((720, 1280, 3), np.uint8)

            # Create HSV tracker Window
            cv2.namedWindow('HSV')
            cv2.createTrackbar('HL', 'HSV', self.hl, 180, nothing)
            cv2.createTrackbar('SL', 'HSV', self.sl, 255, nothing)
            cv2.createTrackbar('VL', 'HSV', self.vl, 255, nothing)
            cv2.createTrackbar('HU', 'HSV', self.hu, 180, nothing)
            cv2.createTrackbar('SU', 'HSV', self.su, 255, nothing)
            cv2.createTrackbar('VU', 'HSV', self.vu, 255, nothing)
            cv2.createTrackbar('AREA_LIMIT', 'HSV', self.area_limit, 1000, nothing)

            # Create Hough tracker Window
            cv2.namedWindow('HOUGH')
            cv2.createTrackbar('DP', 'HOUGH', self.dp, 20, nothing)
            cv2.createTrackbar('MIN_DIST', 'HOUGH', self.min_dist, 100, nothing)
            cv2.createTrackbar('PARAM1', 'HOUGH', self.param1, 700, nothing)
            cv2.createTrackbar('PARAM2', 'HOUGH', self.param2, 200, nothing)
            cv2.createTrackbar('MIN_RADIUS', 'HOUGH', self.min_radius, 100, nothing)
            cv2.createTrackbar('MAX_RADIUS', 'HOUGH', self.max_radius, 100, nothing)
            
        self.bridge = CvBridge()
        self.pub_blobs = rospy.Publisher("/fuels", PointCloud, queue_size=1)
        self.sub_image = message_filters.Subscriber("/zed/rgb/image_raw_color", Image, queue_size=1)
        self.sub_depth = message_filters.Subscriber("/zed/depth/depth_registered", Image, queue_size=1)
        self.ts = message_filters.TimeSynchronizer([self.sub_image, self.sub_depth], 10)
        self.ts.registerCallback(self.processImage)
        
        rospy.loginfo("Fuel detector initialized.")
        
        if self.isTesting:
            self.window_runner()
            


    def processImage(self, image_msg, depth_msg):
        im = self.bridge.imgmsg_to_cv2(image_msg) # Convert image to cv mat using CVBridge

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) # convert color space of image to HSV
        self.msg = PointCloud() # Initialize message to store ball types

        if not self.isTesting:
            self.find_color(im, depth_msg, cv2.inRange(hsv, np.array([20, 0, 201]), np.array([40, 255, 255]))) # Call object discriminator function
            self.pub_blobs.publish(self.msg)
        else:
            self.find_color(im, depth_msg, cv2.inRange(hsv, np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])))
            self.pub_blobs.publish(self.msg)

    def screenToWorld(self, obj_rect, depth, fov_size, frame_size, camera_elavation):
        _depth_obj = 0.127
        rect_center = tuple([obj_rect[0] + obj_rect[2]/2., obj_rect[1] + obj_rect[3]/2.])
        dist_to_center = tuple([rect_center[0] - (frame_size[0]/2.), -1*rect_center[1] + frame_size[1]/2.])

        azimuth = math.atan(dist_to_center[0] / (0.5 * frame_size[0] / math.tan(fov_size[0] / 2.)))
        inclination = math.atan(dist_to_center[1] / (0.5 * frame_size[1] / math.tan(fov_size[1] / 2.)))

        depth += _depth_obj / 2.

        point_3d = (depth * math.cos(inclination) * math.sin(azimuth),
                    depth * math.cos(inclination) * math.cos(azimuth),
                    depth * math.sin(inclination))
        return point_3d



    def find_color(self, passed_im, depth_msg, mask):
        c_im = passed_im.copy() # Make a copy of image to modify
        if self.isTesting:
            self.contour_image = c_im

        # Perform morphology "open" which performs dilate and expand with a kernal size of 3
        k_size = 3
        kernel = np.ones((k_size,k_size),np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find Contours on mask and process contours
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] # Find contours on masked iamge
        points = []
        approx_contours = []
        
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.area_limit: continue

            # Circles and blobs of circles should beat this condition
            epsilon = 0.005*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            if len(approx) < 20: continue

            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            hFov = 105.0
            camera_elavation = .508
            screen_size = tuple([1280, 720])
            fov_size = tuple([hFov*(math.pi/180.)*(screen_size[0]/screen_size[1]),
                              hFov*(math.pi/180.)*(screen_size[0]/screen_size[1])])
            objRect = cv2.boundingRect(c)
            # print "x: {} y: {}".format(cX, cY)
            # print depth_image[cY][cX]
            point3d = self.screenToWorld(objRect, depth_image[cY][cX], fov_size, screen_size, camera_elavation)
            
            point3d = self.screenToWorld([0, 1, 2, 3], 10, [10, 10], [50 ,30], 10)
            
            points.append(point3d)
            # append the convex hull of the contours to fix ignoring the dimples
            approx_contours.append(cv2.convexHull(c))

        # Generate contour based mask
        mask2 = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(mask2, approx_contours, -1, (255, 255, 255), -1)
        # cv2.imshow('CONTOUR_MASK', mask2)

        # Mask original image and create a grayscale image for hough circles
        h_im = cv2.bitwise_and(c_im, c_im, mask=mask2) # flip to mask if necessary ()
        if self.isTesting:
            self.hough_image = h_im        
        gray = cv2.cvtColor(cv2.cvtColor(h_im, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        # Find hough circles 
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=self.dp, 
                                   minDist=self.min_dist, param1=self.param1, param2=self.param2, 
                                   minRadius=self.min_radius, maxRadius=self.max_radius)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if self.isTesting:
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(h_im, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(h_im, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)         

        for i, point in enumerate(points):
            point32 = Point32()
            point32.x = point[0]
            point32.y = point[1]
            point32.z = point[2]
            self.msg.points.append(point32)

        if self.isTesting and points:
            cv2.drawContours(c_im, approx_contours, -1, (100, 255, 100), 2)
    def window_runner(self):
        while(True):
            cv2.imshow('HSV', cv2.resize(self.contour_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            cv2.imshow('HOUGH', cv2.resize(self.hough_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
            k = cv2.waitKey(10)
            self.hl         = cv2.getTrackbarPos('HL', 'HSV')
            self.sl         = cv2.getTrackbarPos('SL', 'HSV')
            self.vl         = cv2.getTrackbarPos('VL', 'HSV')
            self.hu         = cv2.getTrackbarPos('HU', 'HSV')
            self.su         = cv2.getTrackbarPos('SU', 'HSV')
            self.vu         = cv2.getTrackbarPos('VU', 'HSV')
            self.area_limit = cv2.getTrackbarPos('AREA_LIMIT', 'HSV')
            self.dp         = cv2.getTrackbarPos('DP', 'HOUGH')
            self.min_dist   = cv2.getTrackbarPos('MIN_DIST', 'HOUGH')
            self.param1     = cv2.getTrackbarPos('PARAM1', 'HOUGH')
            self.param2     = cv2.getTrackbarPos('PARAM2', 'HOUGH')
            self.min_radius = cv2.getTrackbarPos('MIN_RADIUS', 'HOUGH')
            self.max_radius = cv2.getTrackbarPos('MAX_RADIUS', 'HOUGH')
            
if __name__ == "__main__":
    rospy.init_node("FuelDetector")
    bd = BlobDetector()
    rospy.spin()
    cv2.destroyAllWindows()
