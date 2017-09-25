#! /usr/bin/env python

import rospy

from nav_msgs.msg import Odometry

def talker():
	pub = rospy.Publisher("px4_odom")
	rospy.init_node("to_odom")
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		pub.publish()
		rate.sleep()

def listener():
	rospy.Subscriber("/px4flow/opt_flow")
	rospy.spin()

def callback(data):
	odom = Odometry()
	odom.header.stamp = data.header.stamp
	odom.twist.vx = data.velocity_x
	odom.twist.vy = data.velocity_y
	pub.publish(odom)


if __name__ = "__main__":
	try:
		talker()
	except rospy.ROSInterruptException:
		pass
