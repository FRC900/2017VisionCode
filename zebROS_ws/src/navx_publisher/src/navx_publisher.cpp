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

static ros::Publisher time_pub;
static ros::Publisher imu_pub;
static ros::Publisher odom_pub;
static ros::Publisher raw_pub;

#define PI 3.14159265

int main(int argc, char** argv)
{
	ros::init(argc, argv, "navx_publisher");

	ros::NodeHandle nh;
	// Set up publishers
	// Raw_pub publishes in the ENU (east north up) orientation
	// instead of NED (north east down)
	time_pub = nh.advertise<navx_publisher::stampedUInt64>("/navx/time", 50);
	imu_pub = nh.advertise<sensor_msgs::Imu>("/navx/imu", 50);
	raw_pub = nh.advertise<sensor_msgs::Imu>("/navx/raw", 50);
	odom_pub = nh.advertise<nav_msgs::Odometry>("/navx/odom", 50);
	ros::Rate loop_time(10);
	navx_publisher::stampedUInt64 timestamp;
	geometry_msgs::Quaternion orientation;
	geometry_msgs::Vector3 linear_accel;
	geometry_msgs::Vector3 linear_vel;
	geometry_msgs::Vector3 angular_vel;
	sensor_msgs::Imu imu_msg;
	sensor_msgs::Imu imu_msg_raw;
	nav_msgs::Odometry odom;


	unsigned long long last_time = 0ULL;
	tf::Quaternion last_rot (tf::Vector3(0.,0.,0.),0.);
	tf::Quaternion rot;

	bool firstrun = true;

	AHRS nx("/dev/ttyACM0");
	while(ros::ok()) {
		timestamp.data = nx.GetLastSensorTimestamp();
		timestamp.header.stamp = ros::Time::now();

		//pull orientation data from NavX
		orientation.x = nx.GetQuaternionX();
		orientation.y = nx.GetQuaternionY();
		orientation.z = nx.GetQuaternionZ();
		orientation.w = nx.GetQuaternionW();
		imu_msg.orientation = orientation;

		orientation.x = nx.GetQuaternionY();
		orientation.y = nx.GetQuaternionX();
		orientation.z = -nx.GetQuaternionZ();
		orientation.w = nx.GetQuaternionW();

		imu_msg_raw.orientation = orientation;


		float grav = 9.81;
		// Pull acceleration data from navx
		linear_accel.x = nx.GetWorldLinearAccelX() * grav;
		linear_accel.y = nx.GetWorldLinearAccelY() * grav;
		linear_accel.z = nx.GetWorldLinearAccelZ() * grav;

		imu_msg.linear_acceleration = linear_accel;

		double yaw = nx.GetYaw() * PI / 180;
		double pitch = nx.GetPitch() * PI / 180;
		double roll = nx.GetRoll() * PI / 180;

		//uncomment this to add gravity back into /navx/raw
		//linear_accel.x = linear_accel.x + sin(roll)*cos(pitch)*grav;
		//linear_accel.y = linear_accel.y + cos(roll)*sin(pitch)*grav;
		//linear_accel.z = linear_accel.z + cos(pitch)*cos(roll)*grav;

		imu_msg_raw.linear_acceleration = linear_accel;




		linear_vel.x = nx.GetVelocityX();
		linear_vel.y = nx.GetVelocityY();
		linear_vel.z = nx.GetVelocityZ();



		tf::Quaternion pose;
		tf::quaternionMsgToTF(orientation, pose);
		if(firstrun) last_rot = pose;
		rot = pose * last_rot.inverse();
		tf::Matrix3x3(rot).getRPY(roll,pitch,yaw);
		float time = timestamp.data - last_time;
		angular_vel.x = pitch / time;
		angular_vel.y = -roll / time;
		angular_vel.z = -yaw / time;
		imu_msg.angular_velocity = angular_vel;
		imu_msg_raw.angular_velocity = angular_vel;
		last_rot = pose;
		last_time = timestamp.data;

		firstrun = false;

		//pull position data (all this is is the integral of velocity so it's not very good)
		odom.pose.pose.position.x = nx.GetDisplacementX();
		odom.pose.pose.position.y = nx.GetDisplacementY();
		odom.pose.pose.position.z = nx.GetDisplacementZ();
		odom.pose.pose.orientation = orientation;

		odom.twist.twist.linear = linear_vel;
		odom.twist.twist.angular = angular_vel;

		//set the header of the odometry structure
		odom.header.stamp = ros::Time::now();
		odom.header.frame_id = "nav_current_frame";

		imu_msg.linear_acceleration_covariance = {0,0,0,0,0,0,0,0,0};
		imu_msg.angular_velocity_covariance = {0,0,0,0,0,0,0,0,0};
		imu_msg.orientation_covariance = {0,0,0,0,0,0,0,0,0};


		imu_msg_raw.linear_acceleration_covariance = {0,0,0,0,0,0,0,0,0};
		imu_msg_raw.angular_velocity_covariance = {0,0,0,0,0,0,0,0,0};
		imu_msg_raw.orientation_covariance = {0,0,0,0,0,0,0,0,0};

		odom.twist.covariance = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
		odom.pose.covariance = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

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

		imu_msg_raw.header.frame_id = "navx_base_frame";
		imu_msg.header.frame_id = "navx_base_frame";

		imu_msg_raw.linear_acceleration_covariance = imu_msg.linear_acceleration_covariance;
		imu_msg_raw.angular_velocity_covariance = imu_msg.angular_velocity_covariance;
		imu_msg_raw.orientation_covariance = imu_msg.orientation_covariance;

		imu_msg_raw.header.stamp = ros::Time::now();
		imu_msg.header.stamp = ros::Time::now();
		
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
