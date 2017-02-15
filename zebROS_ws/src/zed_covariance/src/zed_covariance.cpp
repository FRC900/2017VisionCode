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
ofstream out;

class CalcCovariance {
	public:
		CalcCovariance() {  }
		void addMeasurement(vector<double> measurement) {
			Mat m_mat = Mat (measurement);
			//reshape from a 1x3 matrix to a 3x1 matrix
			samples.push_back(m_mat.reshape(0,1));
		}

		void calcCovariance() {
			//calculate the covariance matrix and store it in the class's variable
			cv::calcCovarMatrix(samples, cov, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
		}

		cv::Mat getCovariance() { return cov; }
		
		int numMeasurements() { return samples.size().height; }
	
	private:
		Mat samples;
		Mat cov; Mat mean;
};

static CalcCovariance *poseCov;
static CalcCovariance *twistCov;

void callback(const nav_msgs::Odometry::ConstPtr &odom)
{
	//grab data from the odometry message
	vector<double> pose_m;
	pose_m.push_back(odom->pose.pose.position.x);
	pose_m.push_back(odom->pose.pose.position.y);
	pose_m.push_back(odom->pose.pose.position.z);
	
	//convert quaternion to euler angles
	tf::Quaternion q;
	tf::quaternionMsgToTF(odom->pose.pose.orientation, q);
	tf::Matrix3x3 m(q);
	double roll, pitch, yaw;
	m.getRPY(roll, pitch, yaw);

	//push euler angles to measurements
	pose_m.push_back(roll);
	pose_m.push_back(pitch);
	pose_m.push_back(yaw);

	//add the measurement to the covariance class
	poseCov->addMeasurement(pose_m);
	
	cout << (int)((poseCov->numMeasurements() / 500.0) * 100) << " percent complete" << '\r' << flush;
	//every 500 measurements write the covariance matrix to file
	if(poseCov->numMeasurements() % 500 == 0 && poseCov->numMeasurements() >=50 ) {
		poseCov->calcCovariance();
		Mat cov = poseCov->getCovariance();
    	for(int i = 0; i < cov.rows; i++)
    	{
        	const double* oi = cov.ptr<double>(i);
        	for(int j = 0; j < cov.cols; j++) {
            		if(isnan(oi[j]))
						out << 0 << endl;
					else
						out << oi[j] << endl;
			}
    	}
		cout << "Written to file with " << poseCov->numMeasurements() << " measurements" << endl;

	}

}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "goal_detect");
	ros::NodeHandle nh;

	out.open("zed_calib.dat");

	poseCov = new CalcCovariance();
	twistCov = new CalcCovariance();

	ros::Subscriber sub = nh.subscribe("zed/odom", 1000, callback);

	// Set up publisher
	//pub = nh.advertise<goal_detection::GoalDetection>("goal_detect_msg", 10);

	ros::spin();
	return 0;
}
