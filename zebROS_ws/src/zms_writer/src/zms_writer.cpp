#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/TransformStamped.h>

#include <cv_bridge/cv_bridge.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "zmsout.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

ZMSOut *zmsOut;
void callback(const ImageConstPtr& frameMsg, const ImageConstPtr& depthMsg)
{
	cv_bridge::CvImagePtr cvFrame = cv_bridge::toCvCopy(frameMsg, sensor_msgs::image_encodings::BGR8);
	cv_bridge::CvImagePtr cvDepth = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);

	Mat frame(cvFrame->image.clone());
	pyrDown(frame, frame);
	Mat depth(cvDepth->image.clone());
	pyrDown(depth, depth);
	zmsOut->saveFrame(frame, depth);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "goal_detect");

	char name[PATH_MAX];
	int index = -1;
	int rc;
	struct stat statbuf;
	do
	{
		sprintf(name, "/mnt/900_2/cap%d_0.zms", ++index);
		rc = stat(name, &statbuf);
	}
	while (rc == 0);

	sprintf(name, "/mnt/900_2/cap%d.zms", index);
	zmsOut = new ZMSOut(name, 1, 250, false);

	ros::NodeHandle nh;
	message_filters::Subscriber<Image> frame_sub(nh, "/zed_goal/left/image_raw_color", 20);
	message_filters::Subscriber<Image> depth_sub(nh, "/zed_goal/depth/depth_registered", 20);

	typedef sync_policies::ApproximateTime<Image, Image > MySyncPolicy;
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), frame_sub, depth_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2));

	ros::spin();

	delete zmsOut;

	return 0;
}
