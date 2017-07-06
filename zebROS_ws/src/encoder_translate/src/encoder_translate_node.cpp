#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <nav_msgs/Odometry.h>
  
ros::Publisher odom_pub;

void callback(const geometry_msgs::PointStamped::ConstPtr &msg)
{
	ros::Time current_time, last_time;
	current_time = ros::Time::now();
	last_time = ros::Time::now();

    //first, we'll publish the transform over tf
	geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = current_time;
    odom_trans.header.frame_id = "odom";
    odom_trans.child_frame_id = "base_link";

    odom_trans.transform.translation.x = msg->point.x;
    odom_trans.transform.translation.y = msg->point.y;
    odom_trans.transform.translation.z = 0.0;

	geometry_msgs::Quaternion odom_quat;
    odom_trans.transform.rotation = odom_quat;

    //send the transform
	tf::TransformBroadcaster odom_broadcaster;
    odom_broadcaster.sendTransform(odom_trans);

    //next, we'll publish the odometry message over ROS
    nav_msgs::Odometry odom;
    odom.header.stamp = msg->header.stamp;
    odom.header.frame_id = "odom";
    odom.child_frame_id = "base_link";

    //set the position
    odom.pose.pose.position.x = msg->point.x;
    odom.pose.pose.position.y = msg->point.y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = odom_quat;

#if 0
    //set the velocity
    odom.twist.twist.linear.x = vx;
    odom.twist.twist.linear.y = vy;
    odom.twist.twist.angular.z = vth;
#endif

    //publish the message
    odom_pub.publish(odom);
}

int main(int argc, char** argv){
  ros::init(argc, argv, "encoder_translate");

  ros::NodeHandle n;
  odom_pub = n.advertise<nav_msgs::Odometry>("odom", 50);
  ros::Subscriber sub = n.subscribe("/encoder", 50, callback);

  ros::spin();

}
