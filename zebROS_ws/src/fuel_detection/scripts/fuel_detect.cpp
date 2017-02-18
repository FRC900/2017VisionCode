#include <iostream>
#include <opencv2/opencv.hpp>

#include "FuelDetector.hpp"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/TransformStamped.h>
#include "navx_publisher/stampedUInt64.h"

#include <geometry_msgs/Point32.h>
#include <cv_bridge/cv_bridge.h>

#include "goal_detection/GoalDetection.h"

#include <sstream>

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

static ros::Publisher pub;
static GoalDetector *gd;
static bool batch = true;

class FuelDetector {

	public:
		FuelDetector() {
			bool isTesting = false;
			int hl = 20;
			int sl = 137;
			int vl = 80;
			int hu = 149;
			int su = 255;
			int vu = 255;
			int area_limit = 391;

			int dp = 2;
			int min_dist = 9;
			int param1 = 246;
			int param2 = 48;
			int min_radius = 5;
			int max_radius = 27;
			float contour_image = [[[0]]];
			float hough_image = [[[0]]];
		}

	void createWindows () {
		namedWindow("HSV", WSV);
		createTrackbar("HL", "HSV", this.hl, 180);
        	createTrackbar("SL", "HSV", this.sl, 255);
          	createTrackbar("VL", "HSV", this.vl, 255);
         	createTrackbar("HU", "HSV", this.hu, 180);
            	createTrackbar("SU", "HSV", this.su, 255);
            	createTrackbar("VU", "HSV", this.vu, 255);
            	createTrackbar("AREA_LIMIT", "HSV", this.area_limit, 1000);
	}

}

int main(int argc, char* argv[]) {

	fd = FuelDetector()

	if (argc == 2) {
		this.isTesting = true;	
	}

}
