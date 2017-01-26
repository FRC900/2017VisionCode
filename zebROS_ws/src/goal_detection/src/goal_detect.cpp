#include <iostream>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>

#include "GoalDetector.hpp"
#include "Utilities.hpp"
#include "track3d.hpp"
#include "frameticker.hpp"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float64MultiArray.h"

#include <sstream>

#include "goal_detection/GoalDetection.h"

using namespace cv;
using namespace std;

class SubscribeAndPublish
{
public:
  SubscribeAndPublish()
  {
    //Topic you want to publish
    pub_ = n.advertise<goal_detection::GoalDetection>("pub_msg", 1);

    //wait for messages from ZED wrapper
    //in order to recieve messages from camera_info we have to get a message from image_rect_color first
    ros::topic::waitForMessage("/zed/left/image_rect_color");

    //Topic you want to subscribe
    sub_ = n.subscribe("/zed/left/image_rect_color", 1, &SubscribeAndPublish::callback, this);
    sub_ = n.subscribe("/zed/left/camera_info", 1, &SubscribeAndPublish::initInfo, this);

    //Initialize GoalDetector object
    gd = new GoalDetector(camParams.fov, , !args.batchMode);
  }

void callback(const std_msgs::sensor_msgs/Image msg) 
{
	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	gd.findBoilers(frame,depth);

	goal_detection::GoalDetection pub_msg;
	pub_msg.location.x = gd.goal_pos.x;
	pub_msg.location.y = gd.goal_pos.y;
	pub_msg.location.z = gd.goal_pos.z;
	pub_msg.distance = gd.dist_to_goal;
	pub_msg.angle = gd.angle_to_goal;

	goal_pub.publish(pub_msg);
}

void initInfo(const sensor_msgs::CameraInfo msg) {
	float fx = msg.P[0];
	float fy = msg.P[5];
	float fov_x = 2.0 * atanf(
}

private:
  ros::NodeHandle n; 
  ros::Publisher pub_;
  ros::Subscriber sub_;
  
  GoalDetector gd;
  cv::Point2f fov_size;
  Mat frame_;
};


int main(int argc, char **argv)
{
  //Initiate ROS
  ros::init(argc, argv, "goal_detection");

  //Create an object of class SubscribeAndPublish that will take care of everything
  SubscribeAndPublish SAPObject;

  ros::spin();

  return 0;
}
