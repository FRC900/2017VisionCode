#include <iostream>

#include <ros/ros.h>
#include <sstream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "geometry_msgs/Quaternion.h"
#include "sensor_msgs/Imu.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/Vector3.h"
#include "navXTimeSync/AHRS.h"
#include "navx_publisher/stampedUInt64.h"
#include <tf/transform_datatypes.h>

using namespace cv;
using namespace std;
using namespace sensor_msgs;

static ros::Publisher pub;

class CalcCovariance {
	public:
		CalcCovariance() {}
		void addMeasurement(vector<double> measurement) {
			samples.push_back(measurement);	
		}
		void calcCovariance() {
			Mat in_mat(samples[0].size(), samples.size(), CV_64FC1);
			cout << "Size of samples: " << samples.size() << " " << samples[0].size() << endl;
			for(size_t i = 0; i < samples[0].size(); i++)
				for(size_t j = 0; j < samples.size(); j++)
					in_mat.at<double>(j,i) = samples[j][i];
			//cout << "calcCovar input: " << in_mat << endl;
			cv::transpose(in_mat,in_mat);
			cv::calcCovarMatrix(in_mat, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
		}
		cv::Mat getCovariance() { return cov; }
		int numMeasurements() { return samples.size(); }
	private:
		vector<vector<double>> samples;
		Mat cov; Mat mean;
};

static CalcCovariance *poseCov;
static CalcCovariance *twistCov;

void callback(const nav_msgs::Odometry::ConstPtr &odom)
{
	vector<double> pose_m;
	pose_m.push_back(odom->pose.pose.position.x);
	pose_m.push_back(odom->pose.pose.position.y);
	pose_m.push_back(odom->pose.pose.position.z);
	poseCov->addMeasurement(pose_m);
	if(poseCov->numMeasurements() % 50 == 0 && poseCov->numMeasurements() >=50 ) {
		poseCov->calcCovariance();
		Mat cov = poseCov->getCovariance();
		cout << "Calculated Covariance: " << cov << endl;
	}
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "goal_detect");
	ros::NodeHandle nh;

	poseCov = new CalcCovariance();
	twistCov = new CalcCovariance();

	ros::Subscriber sub = nh.subscribe("zed/odom", 1000, callback);

	// Set up publisher
	//pub = nh.advertise<goal_detection::GoalDetection>("goal_detect_msg", 10);

	ros::spin();
	return 0;
}
