#include <iostream>

#include <ros/ros.h>
#include <sstream>

#include "std_msgs/UInt64.h"
#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/Vector3.h"
#include "navXTimeSync/AHRS.h"
#include "navx_publisher/stampedUInt64.h"
#include "nav_msgs/Odometry.h"

using namespace std;

static ros::Publisher time_pub;
static ros::Publisher imu_pub;
static ros::Publisher odom_pub;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "navx_publisher");

	ros::NodeHandle nh;
	// Set up publisher
	time_pub = nh.advertise<navx_publisher::stampedUInt64>("/navx/time", 100);
	imu_pub = nh.advertise<sensor_msgs::Imu>("/navx/imu", 100);
	odom_pub = nh.advertise<nav_msgs::Odometry>("/navx/odom",100);

	ros::Rate loop_time(10);

	navx_publisher::stampedUInt64 timestamp;
	geometry_msgs::Quaternion orientation;
	geometry_msgs::Vector3 linear_accel;
	geometry_msgs::Vector3 linear_vel;
	geometry_msgs::Vector3 angular_accel;
	geometry_msgs::Vector3 angular_vel;
	sensor_msgs::Imu imu_msg;
	nav_msgs::Odometry odom;

	AHRS nx("/dev/ttyACM0");
	
	while(ros::ok()) {
		timestamp.data = nx.GetLastSensorTimestamp();
		timestamp.header.stamp = ros::Time::now();
		imu_msg.header.stamp = ros::Time::now();
		odom.header.stamp = ros::Time::now();

		orientation.x = nx.GetQuaternionX();
		orientation.y = nx.GetQuaternionY();
		orientation.z = nx.GetQuaternionZ();
		orientation.w = nx.GetQuaternionW();
		imu_msg.orientation = orientation;

		linear_accel.x = nx.GetWorldLinearAccelX();
		linear_accel.y = nx.GetWorldLinearAccelY();
		linear_accel.z = nx.GetWorldLinearAccelZ();
		imu_msg.linear_acceleration = linear_accel;

		linear_vel.x = nx.GetVelocityX();
		linear_vel.y = nx.GetVelocityY();
		linear_vel.z = nx.GetVelocityZ();

		//set angular_accel.x,y,z
		imu_msg.angular_velocity = angular_vel;
		//set angular_vel.x,y,z


		odom.pose.pose.position.x = nx.GetDisplacementX();
		odom.pose.pose.position.y = nx.GetDisplacementY();
		odom.pose.pose.position.z = nx.GetDisplacementZ();
		odom.pose.pose.orientation = orientation;

		odom.twist.twist.linear = linear_vel;
		odom.twist.twist.angular = angular_vel;


		imu_msg.header.stamp = ros::Time::now();
		time_pub.publish(timestamp);
		imu_pub.publish(imu_msg);
		odom_pub.publish(odom);

		ros::spinOnce();
		loop_time.sleep();
	}

	return 0;
}
