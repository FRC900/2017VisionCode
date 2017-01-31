#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
import rospy

from sensor_msgs.msg import Image
from fuel_detection.msg import FuelDetection
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

import math
import time
import sys
import ThreadingUtilities

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
            self.window_thread = ThreadingUtilities.StoppableThread(target=self.window_runner)
            self.window_thread.start()
        
        self.bridge = CvBridge()
        self.pub_blobs = rospy.Publisher("/fuels", FuelDetection, queue_size=1)
        self.sub_image = rospy.Subscriber("/zed/rgb/image_raw_color", Image, self.processImage, queue_size=1)
       
        rospy.loginfo("BlobDetector initialized.")

    def processImage(self, image_msg):
        im = self.bridge.imgmsg_to_cv2(image_msg)
        im = im[:.4*len(im)]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        self.msg = FuelDetection()
        if not self.isTesting:
            self.find_color(im, "green", cv2.inRange(hsv, np.array([45, 110, 100]), np.array([65, 210, 150])))   # green
            if len(self.msg.heights) > 0:
                self.pub_blobs.publish(self.msg)
        else:
            self.find_color(im, "testing",cv2.inRange(hsv, np.array([self.hl, self.sl, self.vl]), np.array([self.hu, self.su, self.vu])))

    def find_color(self, passed_im, label_color, mask):
        im = passed_im.copy()
        if self.isTesting:
            self.image = im
	    # mask = cv2.convertScaleAbs(mask)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        approx_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            perim = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .005*perim, True)
            if len(approx) == 20:
                approx_contours.append(approx)
                # self.msg.areas.append(area)
                self.msg.colors.append(label_color)
                moments = cv2.moments(c)
                center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
                msg_loc = Point()
                msg_loc.x, msg_loc.y = float(center[0]) / len(im[0]), float(center[1]) / len(im)
                self.msg.locations.append(msg_loc)
                self.msg.heights.append(float((max(approx, key=lambda x: x[0][1])[0][1] - min(approx, key=lambda x: x[0][1])[0][1])) / len(im))
                cv2.putText(im, label_color, center, cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 100))
                print "Label color:  {}".format(label_color)
        if approx_contours:
            if self.isTesting:
                cv2.drawContours(self.image, approx_contours, -1, (100, 255, 100), 2)
    def window_runner(self):
        cv2.imshow('HSV', cv2.resize(self.image, Size(), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        k = cv2.waitKey(1) & 0xFF
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
