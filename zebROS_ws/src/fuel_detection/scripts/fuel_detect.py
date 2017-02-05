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

class BlobDetector:
    def __init__(self):
        self.isTesting = False
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
            self.test = self.window_runner() 
        
        self.bridge = CvBridge()
        self.pub_blobs = rospy.Publisher("/fuels", PointCloud, queue_size=1)
        self.sub_image = message_filters.Subscriber("/zed/rgb/image_raw_color", Image, queue_size=1)
        self.sub_depth = message_filters.Subscriber("/zed/depth/depth_registered", Image, queue_size=1)
        self.ts = message_filters.TimeSynchronizer([self.sub_image, self.sub_depth], 10)
        self.ts.registerCallback(self.processImage)
        
        rospy.loginfo("BlobDetector initialized.")

    def processImage(self, image_msg, depth_msg):
        im = self.bridge.imgmsg_to_cv2(image_msg) # Convert image to cv mat using CVBridge
        # im = im[:.4*len(im)]
	
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV) # convert color space of image to HSV
        self.msg = PointCloud() # Initialize message to store ball types

        if not self.isTesting:
            self.find_color(im, depth_msg, cv2.inRange(hsv, np.array([20, 0, 201]), np.array([40, 255, 255]))) # Call object discriminator function
            self.pub_blobs.publish(self.msg)
        else:
            self.find_color(im, depth_msg, cv2.inRange(hsv, np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])))
    
    def screenToWorld(self, obj_rect, depth, fov_size, frame_size, camera_elavation):
        _depth_obj = 0.127
        rect_center = tuple((obj_rect[0] + obj_rect[2]/2), (obj_rect[1] + obj_rect[3]/2))
        dist_to_center = tuple((rect_center[0] - (frame_size[0]/2)), (-1*rect_center[1] + frame_size[1]/2))
        
        azimuth = math.atan2(dist_to_center[0] / (0.5 * frame_size[0] / math.tan(fov_size[0] / 2)))
        inclination = math.atan2(dist_to_center[1] / (0.5 * frame_size[1] / math.tan(fov_size[1] / 2)))
        
        depth += _depth_obj / 2.
        
        point_3d = (depth * math.cos(inclination) * math.sin(azimuth),
                    depth * math.cos(inclination) * math.cos(azimuth),
                    depth * math.sin(inclination))
        return point_3d
        
    
    
    def find_color(self, passed_im, depth_msg, mask):
        im = passed_im.copy() # Make a copy of image to modify
        if self.isTesting:
            self.image = im
	    # mask = cv2.convertScaleAbs(mask)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1] # Find contours on masked iamge
        points = []
        approx_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 100: continue
            # perim = cv2.arcLength(c, True)
            # approx = cv2.approxPolyDP(c, .005*perim, True)
            # if len(approx) == 20:
            # approx_contours.append(approx)
            moments = cv2.moments(c)
            center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
            c_center = (float(center[0]) / len(im[0]), float(center[1]) / len(im))
            
            hFov = 105.0
            camera_elavation = .508
            screen_size = (1280, 720)
            fov_size = tuple(
                            (hFov * (math.pi / 180.) * (screen_size[0] / screen_size[1]), 
                            (hFov * (math.pi / 180.) * (screen_size[0] / screen_size[1])
                            )
            objRect = cv2.BoundingRect(c)
            point3d = screenToWorld(objRect, depth_msg.data[c_center[0]][c_center[1]], fov_size, screen_size, camera_elavation)
            
            points.append(point3d)
            approx_contours.append(c)
        # self.msg.points.resize(len(points))
        for i, point in enumerate(points):
            point = Point32()
            point.x = point[0]
            point.y = point[1]
            point.z = point[2]
            self.msg.points.append(point)

       
            # self.msg.locations.append(msg_loc)
            # self.msg.heights.append(float((max(approx, key=lambda x: x[0][1])[0][1] - min(approx, key=lambda x: x[0][1])[0][1])) / len(im))
            # cv2.putText(im, label_color, center, cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 100))
            # print "Label color:  {}".format(label_color)
        
        if points:
            if self.isTesting:
                cv2.drawContours(self.image, approx_contours, -1, (100, 255, 100), 2)
    def window_runner(self):
        while(True):
            cv2.imshow('HSV', self.image)
            k = cv2.waitKey(10)
            if k == 27:
                self.window_thread.stop()
            self.hl = cv2.getTrackbarPos('HL', 'HSV')
            self.sl = cv2.getTrackbarPos('SL', 'HSV')
            self.vl = cv2.getTrackbarPos('VL', 'HSV')
            self.hu = cv2.getTrackbarPos('HU', 'HSV')
            self.su = cv2.getTrackbarPos('SU', 'HSV')
            self.vu = cv2.getTrackbarPos('VU', 'HSV')

if __name__ == "__main__":
    rospy.init_node("BlobDetector")
    bd = BlobDetector()
    rospy.spin()
    cv2.destroyAllWindows()
