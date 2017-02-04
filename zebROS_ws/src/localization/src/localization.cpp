#include <iostream>

#include <ros/ros.h>
#include <sstream>

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

static ros::Publisher pub;



void callback(const sensor_msgs::ImuConstPtr& imuMsg, const nav_msgs::OdometryConstPtr& odomMsg, const navx_publisher::stampedUInt64ConstPtr& timeMsg)
{



	//Do stuf


}



int main(int argc, char** argv)
{
	ros::init(argc, argv, "localization");

	ros:NodeHandle nh;
	message_filters::Subscriber<sensor_msgs::Imu> imu(nh, "/navx/imu", 10);
	message_filters::Subscriber<nav_msgs::Odometry> odom(nh, "/navx/odom", 10);
	message_filters::Subscriber<navx_publisher::stampedUInt64> time(nh, "/navx/time", 10);

	typedef sync_policies::ApproximateTime<sensor_msgs::Imu,odom,_msg,navx_publisher::stampedUInt64> syncPolicy;
	Synchronizer<syncPolicy> sync(syncPolicy(50), imu, odom, time);
	sync.registerCallback(boost:bind(&callback, _1, _2, _3));

	localization::LocalizationMessage local_msg;

	pub = nh.advertise<localization:LocalizationMessage>("/localization/pos", 10);

	ros::spin();

	return 0;
	
}
