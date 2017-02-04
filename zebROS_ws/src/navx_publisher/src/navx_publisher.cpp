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
	sensor_msgs::Imu imu_msg;

	AHRS nx("/dev/ttyACM0");
	while(ros::ok()) {
		timestamp.data = nx.GetLastSensorTimestamp();
		timestamp.header.stamp = ros::Time::now();

		orientation.x = nx.GetQuaternionX();
		orientation.y = nx.GetQuaternionY();
		orientation.z = nx.GetQuaternionZ();
		orientation.w = nx.GetQuaternionW();
		imu_msg.orientation = orientation;

		linear_accel.x = nx.GetWorldLinearAccelX();
		linear_accel.y = nx.GetWorldLinearAccelY();
		linear_accel.z = nx.GetWorldLinearAccelZ();
		imu_msg.linear_acceleration = linear_accel;

		imu_msg.header.stamp = ros::Time::now();
		time_pub.publish(timestamp);
		imu_pub.publish(imu_msg);

		ros::spinOnce();
		loop_time.sleep();
	}

	return 0;
}
