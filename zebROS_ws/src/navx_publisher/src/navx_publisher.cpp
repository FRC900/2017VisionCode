#include <iostream>

#include <ros/ros.h>
#include <sstream>

#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/Vector3.h"
#include "navXTimeSync/AHRS.h"
#include "navx_publisher/stampedUInt64.h"

using namespace std;

static ros::Publisher time_pub;
static ros::Publisher imu_pub;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "navx_publisher");

	ros::NodeHandle nh;
	// Set up publisher
	time_pub = nh.advertise<navx_publisher::stampedUInt64>("/navx/time", 50);
	imu_pub = nh.advertise<sensor_msgs::Imu>("/navx/imu", 50);
	ros::Rate loop_time(20);
	navx_publisher::stampedUInt64 timestamp;
	geometry_msgs::Quaternion orientation;
	geometry_msgs::Vector3 linear_accel;
	geometry_msgs::Vector3 linear_vel;
	geometry_msgs::Vector3 angular_vel;
	sensor_msgs::Imu imu_msg;


	navx_publisher::stampedUInt64 last_time;
	tf::Vector3 last_rot;
	tf::Vector3 rot;

	AHRS nx("/dev/ttyACM0");
	while(ros::ok()) {
		timestamp.data = nx.GetLastSensorTimestamp();
		timestamp.header.stamp = ros::Time::now();

		orientati+on.x = nx.GetQuaternionX();
		orientation.y = nx.GetQuaternionY();
		orientation.z = nx.GetQuaternionZ();
		orientation.w = nx.GetQuaternionW();
		imu_msg.orientation = orientation;

		float grav = 9.81;

		linear_accel.x = nx.GetWorldLinearAccelX() * grav;
		linear_accel.y = nx.GetWorldLinearAccelY() * grav;
		linear_accel.z = nx.GetWorldLinearAccelZ() * grav;
		imu_msg.linear_acceleration = linear_accel;

		linear_vel.x = nx.GetVelocityX();
		linear_vel.y = nx.GetVelocityY();
		linear_vel.z = nx.GetVelocityZ();

		tf::Quaternion pose;
		tf::quaternionMsgToTF(orientation, pose);
		rot = orientation * last_rot.inverse();
		double roll, pitch, yaw;
		tf::Matrix3x3(rot).getRPY(roll,pitch,yaw);
		float time = timestamp.data - last_time;
		angular_vel.x = roll / time;
		angular_vel.y = pitch / time;
		angular_vel.z = yaw / time;
		imu_msg.angular_velocity = angular_vel;
		last_rot = rot;
		last_time = timestamp.data;

		
		odom.pose.pose.position.x = nx.GetDisplacementX();
		odom.pose.pose.position.y = nx.GetDisplacementY();
		odom.pose.pose.position.z = nx.GetDisplacementZ();
		odom.pose.pose.orientation = orientation;

		odom.twist.twist.linear = linear_vel;
		odom.twist.twist.angular = angular_vel;

		
		imu_msg.angular_velocity_covariance = [0,0,0,0,0,0,0,0,0];

		imu_msg.header.stamp = ros::Time::now();
		time_pub.publish(timestamp);
		imu_pub.publish(imu_msg);

		ros::spinOnce();
		loop_time.sleep();
	}

	return 0;
}
