#include <iostream>
#include <opencv2/opencv.hpp>

#include "GoalDetector.hpp"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/TransformStamped.h>

#include <geometry_msgs/Point32.h>
#include <cv_bridge/cv_bridge.h>

#include <sstream>

#include "GoalDetector.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;;

static ros::Publisher pub;
static GoalDetector *gd;
static bool batch = false;

void callback(const ImageConstPtr& frameMsg, const ImageConstPtr& depthMsg)
{
	cv_bridge::CvImagePtr cvFrame = cv_bridge::toCvCopy(frameMsg, sensor_msgs::image_encodings::BGR8);
	cv_bridge::CvImagePtr cvDepth = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);

	// Maybe pyrdown both inputs for speed?

	gd->findBoilers(cvFrame->image, cvDepth->image);

	geometry_msgs::Point32 point_msg;
	Point3f pt = gd->goal_pos();
	point_msg.x = pt.x;
	point_msg.y = pt.y;
	point_msg.z = pt.z;

	pub.publish(point_msg);

	if (!batch)
	{
		Mat frame(cvFrame->image.clone());
		gd->drawOnFrame(frame, gd->getContours(cvFrame->image));
		cout << "Size : " << cvFrame->image.size() << endl;
		pyrDown(frame, frame);
		imshow("Image", frame);
		waitKey(5);
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "goal_detect");

	ros::NodeHandle nh;
	message_filters::Subscriber<Image> frame_sub(nh, "/zed/left/image_raw_color", 1);
	message_filters::Subscriber<Image> depth_sub(nh, "/zed/depth/depth_registered", 1);

	typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
	// ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
	Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), frame_sub, depth_sub);
	sync.registerCallback(boost::bind(&callback, _1, _2));

	// Create goal detector class
	const float hFov = 105.;
	const Size size(1280, 720);
	const Point2f fov(hFov * (M_PI / 180.), hFov * (M_PI / 180.) * ((float)size.height / size.width));
	gd = new GoalDetector(fov, size, true);

	// Set up publisher
	pub = nh.advertise<geometry_msgs::Point32>("goal_detect_msg", 1);

	ros::spin();

	delete gd;

	return 0;
}
