#include <iostream>

#include <ros/ros.h>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>

#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Vector3.h"
#include "navXTimeSync/AHRS.h"
#include "navx_publisher/stampedUInt64.h"
#include <tf/transform_datatypes.h>

using namespace std;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "navx_publisher");

	ros::NodeHandle nh;
	// Set up publishers
	// Raw_pub publishes in the ENU (east north up) orientation
	// instead of NED (north east down)
	ros::Publisher time_pub = nh.advertise<navx_publisher::stampedUInt64>("/navx/time", 50);
	ros::Publisher imu_pub = nh.advertise<sensor_msgs::Imu>("/navx/imu", 50);
	ros::Publisher raw_pub = nh.advertise<sensor_msgs::Imu>("/navx/raw", 50);
	ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/navx/odom", 50);
	ros::Rate loop_time(15);
	navx_publisher::stampedUInt64 timestamp;
	sensor_msgs::Imu imu_msg;
	sensor_msgs::Imu imu_msg_raw;
	nav_msgs::Odometry odom;

	imu_msg.linear_acceleration_covariance = {0,0,0,0,0,0,0,0,0};
	imu_msg.angular_velocity_covariance = {0,0,0,0,0,0,0,0,0};
	imu_msg.orientation_covariance = {0,0,0,0,0,0,0,0,0};


	imu_msg_raw.linear_acceleration_covariance = {0,0,0,0,0,0,0,0,0};
	imu_msg_raw.angular_velocity_covariance = {0,0,0,0,0,0,0,0,0};
	imu_msg_raw.orientation_covariance = {0,0,0,0,0,0,0,0,0};

	odom.twist.covariance = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	odom.pose.covariance = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	imu_msg_raw.header.frame_id = "navx_frame";
	imu_msg.header.frame_id = "navx_frame";
	odom.header.frame_id = "nav_current_frame";

	{
		//read the file with covariances and apply it to the odometry and IMU
		ifstream infile("/home/ubuntu/2017VisionCode/zebROS_ws/src/navx_publisher/navx_calib.dat");
		if(!infile.good())
			cerr << "File not opened!" << endl;
		std::string line;
		int ln = 0;
		while(std::getline(infile, line))
		{
			if(line == "") break;
			imu_msg.linear_acceleration_covariance[ln] = std::stod(line);
			ln++;
		}
		ln = 0;
		while(std::getline(infile, line))
		{
			if(line == "") break;
			imu_msg.angular_velocity_covariance[ln] = std::stod(line);
			ln++;
		}
		ln = 0;
		while(std::getline(infile, line))
		{
			if(line == "") break;
			imu_msg.orientation_covariance[ln] = std::stod(line);
			ln++;
		}
		ln = 0;
		while(std::getline(infile, line))
		{
			if(line == "") break;
			odom.twist.covariance[ln] = std::stod(line);
			odom.pose.covariance[ln] = std::stod(line);
			ln++;
		}
	}
	imu_msg_raw.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance;
	imu_msg_raw.angular_velocity_covariance = imu_msg.angular_velocity_covariance;
	imu_msg_raw.orientation_covariance = imu_msg.orientation_covariance;

	ros::Time last_time;
	tf::Quaternion last_rot (tf::Vector3(0.,0.,0.),0.);
	tf::Quaternion rot;

	bool firstrun = true;

	AHRS nx("/dev/ttyACM0");
	while(ros::ok()) {
		timestamp.data = nx.GetLastSensorTimestamp();

		//set the timestamp for all headers
		odom.header.stamp = 
		imu_msg.header.stamp = 
		imu_msg_raw.header.stamp = 
		timestamp.header.stamp = ros::Time::now();

		//pull orientation data from NavX
		imu_msg.orientation.x = nx.GetQuaternionX();
		imu_msg.orientation.y = nx.GetQuaternionY();
		imu_msg.orientation.z = nx.GetQuaternionZ();
		imu_msg.orientation.w = nx.GetQuaternionW();

		imu_msg_raw.orientation.x = imu_msg.orientation.x;
		imu_msg_raw.orientation.y = imu_msg.orientation.y;
		imu_msg_raw.orientation.z = -imu_msg.orientation.z;
		imu_msg_raw.orientation.w = imu_msg.orientation.w;

		const double grav = 9.80665;
		// Pull acceleration data from navx
		imu_msg.linear_acceleration.x = nx.GetWorldLinearAccelX() * grav;
		imu_msg.linear_acceleration.y = nx.GetWorldLinearAccelY() * grav;
		imu_msg.linear_acceleration.z = nx.GetWorldLinearAccelZ() * grav;

		double yaw = nx.GetYaw() * M_PI / 180.;
		double pitch = nx.GetPitch() * M_PI / 180.;
		double roll = nx.GetRoll() * M_PI / 180.;

#if 1
		imu_msg_raw.linear_acceleration = imu_msg.linear_acceleration;
#else
		//uncomment this to add gravity back into /navx/raw
		imu_msg_raw.linear_acceleration.x = imu_msg.linear_acceleration.x + sin(roll)*cos(pitch)*grav;
		imu_msg_raw.linear_acceleration.y = imu_msg.linear_acceleration.y + cos(roll)*sin(pitch)*grav;
		imu_msg_raw.linear_acceleration.z = imu_msg.linear_acceleration.z + cos(pitch)*cos(roll)*grav;
#endif

		tf::Quaternion pose;
		tf::quaternionMsgToTF(imu_msg_raw.orientation, pose); // or imu_msg? they differ in the z value
		if(firstrun) last_rot = pose;
		rot = pose * last_rot.inverse();
		tf::Matrix3x3(rot).getRPY(roll,pitch,yaw);
		const double dTime = odom.header.stamp.toSec() - last_time.toSec();
		imu_msg.angular_velocity.x = pitch / dTime;
		imu_msg.angular_velocity.y = -roll / dTime;
		imu_msg.angular_velocity.z = -yaw / dTime;
		imu_msg_raw.angular_velocity = imu_msg.angular_velocity;
		last_rot = pose;
		last_time = odom.header.stamp;

		firstrun = false;

		//pull position data (all this is is the integral of velocity so it's not very good)
		odom.pose.pose.position.x = nx.GetDisplacementX();
		odom.pose.pose.position.y = nx.GetDisplacementY();
		odom.pose.pose.position.z = nx.GetDisplacementZ();
		odom.pose.pose.orientation = imu_msg_raw.orientation; // or imu_msg? see above question

		odom.twist.twist.linear.x = nx.GetVelocityX();
		odom.twist.twist.linear.y = nx.GetVelocityY();
		odom.twist.twist.linear.z = nx.GetVelocityZ();

		odom.twist.twist.angular = imu_msg.angular_velocity;
		
		//publish to ROS topics
		time_pub.publish(timestamp);
		imu_pub.publish(imu_msg);
		odom_pub.publish(odom);
		raw_pub.publish(imu_msg_raw);
		ros::spinOnce();
		loop_time.sleep();
	}

	return 0;
}
