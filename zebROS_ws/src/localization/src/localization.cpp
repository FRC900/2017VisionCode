#include <iostream>

#include <ros/ros.h>
#include <sstream>

#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Vector3.h"
#include "navXTimeSync/AHRS.h"
#include "navx_publisher/stampedUInt64.h"
#include "goal_detection/GoalDetection.h"
#include <tf/transform_datatypes.h>


#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/TransformStamped.h>

#include <geometry_msgs/Point32.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

static ros::Publisher pub;



void callback(const sensor_msgs::ImuConstPtr& imuMsg, const nav_msgs::OdometryConstPtr& odomMsg, const navx_publisher::stampedUInt64ConstPtr& timeMsg, const goal_detection::GoalDetectionConstPtr goal)
{

	

	localization::LocalizationMessage lmsg;
	
	//Parse NavX position dat
	Point3f navx_pose = odomMsg.pose.pose.position;


	//Parse goal_detection position data
	Point3f goal_pose = goal.location;

	//Publish data
	pub.publish(lmsg);	


}



int main(int argc, char** argv)
{
	ros::init(argc, argv, "localization");

	ros:NodeHandle nh;
	message_filters::Subscriber<sensor_msgs::Imu> imu(nh, "/navx/imu", 10);
	message_filters::Subscriber<nav_msgs::Odometry> odom(nh, "/navx/odom", 10);
	message_filters::Subscriber<navx_publisher::stampedUInt64> time(nh, "/navx/time", 10);
	message_filters::Subscriber<goal_detection::GoalDetection> gd(nh, "goal_detect_msg", 10);

	typedef sync_policies::ApproximateTime<sensor_msgs::Imu,odom,_msg,navx_publisher::stampedUInt64> syncPolicy;
	Synchronizer<syncPolicy> sync(syncPolicy(50), imu, odom, time);
	sync.registerCallback(boost:bind(&callback, _1, _2, _3));

	localization::LocalizationMessage local_msg;

	pub = nh.advertise<localization:LocalizationMessage>("/localization/pos", 10);

	ros::spin();

	return 0;
	
}
