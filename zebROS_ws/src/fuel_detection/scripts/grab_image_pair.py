#!/usr/bin/env python

# SciStacvk 
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ROS imports
import message_filters
import rospy

# Image Processing
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

class ImageGrab:
	def __init__(self):
		self.bridge = CvBridge()
		self.sub_image = message_filters.Subscriber("/zed/rgb/image_raw_color", Image, queue_size=1)
		self.sub_depth = message_filters.Subscriber("/zed/depth/depth_registered", Image, queue_size=1)
		self.ts = message_filters.TimeSynchronizer([self.sub_image, self.sub_depth], 10)
		self.ts.registerCallback(self.depthSyncCallback)
		
		rospy.loginfo("ImageGrabber initialized.")
	def depthSyncCallback(self, image_msg, depth_msg):
		image = self.bridge.imgmsg_to_cv2(image_msg) # Convert image to cv mat using CVBridge
		depth = self.bridge.imgmsg_to_cv2(depth_msg) # Convert depth to cv mat using CVBridge

		timeStamp = str(rospy.Time.now())

		imageName = timeStamp+"image.png"
		depthName = timeStamp+"depth.png"
		
		cv2.imwrite(imageName, image, (cv2.IMWRITE_PNG_COMPRESSION, 9))
		cv2.imwrite(depthName, depth, (cv2.IMWRITE_PNG_COMPRESSION, 9))

if __name__ == "__main__":
	rospy.init_node("ImageDepthGrabber")
	ig = ImageGrab()
	rospy.spin()
	cv2.destroyAllWindows()