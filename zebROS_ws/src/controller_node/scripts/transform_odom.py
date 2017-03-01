#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry

class TransformOdometry:
    # Pose Position Transform
    X_POS_SHIFT = 
    Y_POS_SHIFT = 
    Z_POS_SHIFT = 
    # Pose Orientation Transform
    X_ORI_SHIFT = 
    Y_ORI_SHIFT = 
    Z_ORI_SHIFT = 
    W_ORI_SHIFT = 

    def qv_mult(q, v):
        v = tf.transformations.unit_vector(v)
        qp = list(v)
        qp.append(0.0)
        return tf.transformations.quaternion_multiply(tf.transformations.quaternion_multiply(q,qp),tf.transformations.quaternion_conjugate(q))

    def __init__(self):
        self.odom_pub_topic = "/odometry"
        self.odom_sub_topic = "/odometry_transformed"

        self.x_pos_shift = 0
        self.y_pos_shift = 0
        self.z_pos_shift = 0

        odom_msg.pose.pose.position.x = odom_msg.pose.pose.position.x / math.cos(Z_ORI_SHIFT) / math.cos(Y_ORI_SHIFT)  
        odom_msg.pose.pose.position.y = odom_msg.pose.pose.position.y / math.sin(Z_ORI_SHIFT) / math.cos(Y_ORI_SHIFT)
        odom_msg.pose.pose.position.z = 0

        self.odom_pub_topic = rospy.get_param("~odom_pub_topic")
        self.odom_sub_topic = rospy.get_param("~odom_sub_topic")

        self.sub_odom_transform = rospy.Subscriber("/zed_fuel/odom", Odometry)
        self.pub_odom_transform = rospy.Publisher("/zed_fuel/odom_transformed", Odometry, queue_size=1)
    
    def _transform_callback(self, odom_msg):
        new_vec = qv_mult(odom_msg.pose.pose.orientation, [X_POS_SHIFT,Y_POS_SHIFT,Z_POS_SHIFT])
        tx = X_POS_SHIFT - new_vec[0]
        ty = Y_POS_SHIFT - new_vec[1]
        tz = Z_POS_SHIFT - new_vec[2]

        odom_pub_msg = Odometry()
        odom_pub_msg.header.seq = odom_msg.header.seq 
        odom_pub_msg.header.stamp = odom_msg.header.stamp
        odom_pub_msg.header.frame_id = odom_msg.header.frame_id
        odom_pub_msg.child_frame_id = "base_link"
        odom_pub_msg.pose.pose.position.x = odom_msg.pose.pose.position.x + tx - X_POS_SHIFT
        odom_pub_msg.pose.pose.position.y = odom_msg.pose.pose.position.y + ty - Y_POS_SHIFT
        odom_pub_msg.pose.pose.position.z = odom_msg.pose.pose.position.z + tz - Z_POS_SHIFT
        odom_pub_msg.pose.pose.orientation.x = odom_msg.pose.pose.orientation.x
        odom_pub_msg.pose.pose.orientation.y = odom_msg.pose.pose.orientation.y
        odom_pub_msg.pose.pose.orientation.z = odom_msg.pose.pose.orientation.z
        odom_pub_msg.pose.pose.orientation.w = odom_msg.pose.pose.orientation.w
        odom_pub_msg.pose.covariance = odom_msg.pose.covariance

        pub_odom_transform.publish(odom_pub_msg)



if __name__ == "__main__":
    to = TransformOdometry()
    rospy.spin()
