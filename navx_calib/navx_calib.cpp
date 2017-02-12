#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>
#include <vector>

#include "AHRS.h"
#include<opencv2/opencv.hpp>

using namespace std;




int main(int argc, char** argv)
{
	AHRS com = AHRS("/dev/ttyACM0");
	cout << "Initializing" << endl << endl;
	cout << "Please ensure navX is perfectly stationary, calibration will begin shortly" << endl << endl;

	this_thread::sleep_for(chrono::milliseconds(1000));

	cout << "Starting calibration... Collecting Data" << endl;

	//Take Initial Values


	vector<double> nrot = {com.GetYaw(), com.GetPitch(), com.GetRoll()};
	vector<double> nlin = {com.GetWorldLinearAccelX(), com.GetWorldLinearAccelY(), com.GetWorldLinearAccelZ()};
	long long int ntime = com.GetLastSensorTimestamp();

	//Init Data Sets

	vector<vector<double>> rot = {nrot};
	vector<vector<double>> lin = {nlin};
	vector<vector<double>> ang;
	vector<long long int> time = {ntime};


	//Collect Data

	for(int i = 1; i < 500; i++)
	{


	
		vector<double> crot = {com.GetYaw(), com.GetPitch(), com.GetRoll()};
		rot.push_back(crot);

		vector<double> clin = {com.GetWorldLinearAccelX(), com.GetWorldLinearAccelY(), com.GetWorldLinearAccelZ()};
		lin.push_back(clin);

		long long int ctime = com.GetLastSensorTimestamp();
		time.push_back(ctime);
	
		vector<double> cang = {(rot[i][0] - rot[i - 1][0])/(time[i] - time[i - 1]),
				     (rot[i][1] - rot[i - 1][1])/(time[i] - time[i - 1]),
				     (rot[i][2] - rot[i - 1][2])/(time[i] - time[i - 1])};
		ang.push_back(cang);


		cout << (i / 5) << "% Complete" << '\r' << flush;
		this_thread::sleep_for(chrono::milliseconds(125));

	
	}

	cout << endl << endl;
	cout << "Data Collection Complete... Calculating Covariance..." << endl;

	//Calculate Covariances


	ofstream out;
	out.open("navx_calib.dat");

	cv::Mat rot_cov;
	cv::Mat rotin(rot[0].size(), rot.size(), CV_64FC1);
	for (size_t i = 0; i < rot[0].size(); i++) 
	{
		for (size_t j = 0; j < rot.size(); j++) {
		    rotin.at<double>(j, i) = rot[j][i];
		}
	}
    	cv::Mat rotout;
    	vector<double> mean;
    	cv::calcCovarMatrix(rotin, rotout, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    	for(int i = 0; i < rotout.rows; i++)
    	{
        	const double* oi = rotout.ptr<double>(i);
        	for(int j = 0; j < rotout.cols; j++)
            		out << oi[j] << endl;
    	}
	out << endl;


	cv::Mat lin_cov;
	cv::Mat linin(lin[0].size(), lin.size(), CV_64FC1);
	for (size_t i = 0; i < lin[0].size(); i++) 
	{
		for (size_t j = 0; j < lin.size(); j++) {
		    linin.at<double>(j, i) = lin[j][i];
		}
	}
    	cv::Mat linout;
    	cv::calcCovarMatrix(linin, linout, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    	for(int i = 0; i < linout.rows; i++)
    	{
        	const double* oi = linout.ptr<double>(i);
        	for(int j = 0; j < linout.cols; j++)
            		out << oi[j] << endl;
    	}
	out << endl;


	cv::Mat ang_cov;
	cv::Mat angin(ang[0].size(), ang.size(), CV_64FC1);
	for (size_t i = 0; i < ang[0].size(); i++) 
	{
		for (size_t j = 0; j < ang.size(); j++) {
		    angin.at<double>(j, i) = ang[j][i];
		}
	}
    	cv::Mat angout;
    	cv::calcCovarMatrix(angin, angout, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    	for(int i = 0; i < angout.rows; i++)
    	{
        	const double* oi = angout.ptr<double>(i);
        	for(int j = 0; j < angout.cols; j++)
            		out << oi[j] << endl;
    	}
	out << endl;


	out.close();

	cout << "Calibration complete, please copy navx_calib.dat to ROS wrapper source folder." << endl;

	return 0;
}


