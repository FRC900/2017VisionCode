#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry

def qv_mult(q, v):
    v = tf.transformations.unit_vector(v)
    qp = list(v)
    qp.append(0.0)
    return tf.transformations.quaternion_multiply(tf.transformations.quaternion_multiply(q,qp),tf.transformations.quaternion_conjugate(q))

class TransformOdometry:
    def __init__(self): 
        self.odom_node_name = "odometry_transform_node"
        self.odom_pub_topic = "/odometry"
        self.odom_sub_topic = "/odometry_transformed"

        self.x_pos_shift = 0
        self.y_pos_shift = 0
        self.z_pos_shift = 0
        
        self.x_ori_shift = 0
        self.y_ori_shift = 0
        self.z_ori_shift = 0
        self.w_ori_shift = 0

        self.odom_node_name = rospy.get_param("~odom_node_name")
        self.odom_pub_topic = rospy.get_param("~odom_pub_topic")
        self.odom_sub_topic = rospy.get_param("~odom_sub_topic")
        
        self.x_pos_shift    = rospy.get_param("~x_pos_shift")
        self.y_pos_shift    = rospy.get_param("~y_pos_shift")
        self.z_pos_shift    = rospy.get_param("~z_pos_shift")

        self.x_ori_shift    = rospy.get_param("~x_ori_shift")
        self.y_ori_shift    = rospy.get_param("~y_ori_shift")
        self.z_ori_shift    = rospy.get_param("~z_ori_shift")
        self.w_ori_shift    = rospy.get_param("~w_ori_shift")

        self.sub_odom_transform = rospy.Subscriber(self.odom_sub_topic, Odometry)
        self.pub_odom_transform = rospy.Publisher(self.odom_pub_topic, Odometry, queue_size=1)

    def get_node_name(self):
        return self.odom_node_name

    def _transform_callback(self, odom_msg):

        odom_msg.pose.pose.position.x = odom_msg.pose.pose.position.x * math.cos(self.z_ori_shift) - odom_msg.pose.pose.position.y * math.sin(self.z_ori_shift)
        odom_msg.pose.pose.position.y = odom_msg.pose.pose.position.x * math.sin(self.z_ori_shift) + odom_msg.pose.pose.posiiton.y * math.cos(self.z_ori_shift)
	
        odom_msg.pose.pose.position.x = odom_msg.pose.pose.position.x * math.cos(self.y_ori_shift) - odom_msg.pose.pose.position.z * math.sin(self.y_ori_shift)
        odom_msg.pose.pose.position.y = odom_msg.pose.pose.position.x * math.sin(self.y_ori_shift) + odom_msg.pose.pose.posiiton.z * math.cos(self.y_ori_shift)

        odom_msg.pose.pose.position.z = 0

        new_vec = qv_mult(odom_msg.pose.pose.orientation, [self.x_pos_shift,self.y_pos_shift,self.z_pos_shift])
        tx = self.x_pos_shift - new_vec[0]
        ty = self.y_pos_shift - new_vec[1]
        tz = self.z_pos_shift - new_vec[2]

        odom_pub_msg = Odometry()
        odom_pub_msg.header.seq = odom_msg.header.seq 
        odom_pub_msg.header.stamp = odom_msg.header.stamp
        odom_pub_msg.header.frame_id = odom_msg.header.frame_id
        odom_pub_msg.child_frame_id = "base_link"
        odom_pub_msg.pose.pose.position.x = odom_msg.pose.pose.position.x + tx - self.x_pos_shift
        odom_pub_msg.pose.pose.position.y = odom_msg.pose.pose.position.y + ty - self.y_pos_shift
        odom_pub_msg.pose.pose.position.z = odom_msg.pose.pose.position.z + tz - self.z_pos_shift
        odom_pub_msg.pose.pose.orientation.x = odom_msg.pose.pose.orientation.x
        odom_pub_msg.pose.pose.orientation.y = odom_msg.pose.pose.orientation.y
        odom_pub_msg.pose.pose.orientation.z = odom_msg.pose.pose.orientation.z
        odom_pub_msg.pose.pose.orientation.w = odom_msg.pose.pose.orientation.w
        odom_pub_msg.pose.covariance = odom_msg.pose.covariance

        pub_odom_transform.publish(odom_pub_msg)

if __name__ == "__main__":
    to = TransformOdometry()
    rospy.init_node(to.get_node_name())
    rospy.spin()
