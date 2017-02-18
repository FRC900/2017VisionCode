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

	vector<vector<double>> rot = {{nrot[0]},{nrot[1]},{nrot[2]}};
	vector<vector<double>> lin = {{nlin[0]},{nlin[1]},{nlin[2]}};
	vector<vector<double>> ang = {{0},{0},{0}};
	vector<long long int> time = {ntime};


	//Collect Data
	int d_points = 500;
	for(int i = 1; i < d_points; i++)
	{


	
		vector<double> crot = {com.GetYaw(), com.GetPitch(), com.GetRoll()};
		rot[0].push_back(crot[0]);
		rot[1].push_back(crot[1]);
		rot[2].push_back(crot[2]);

		vector<double> clin = {com.GetWorldLinearAccelX(), com.GetWorldLinearAccelY(), com.GetWorldLinearAccelZ()};
		//cout << "Gyro: " << com.GetPitch() << " " << com.GetRoll() << " " << com.GetYaw() << endl;
		//cout << "World accelerations: " << clin[0] << " " << clin[1] << " " << clin[2] << endl;
		//cout << "Raw Accelerations: " << com.GetRawAccelX() << " " << com.GetRawAccelY() << " " << com.GetRawAccelZ() << endl;
		lin[0].push_back(clin[0]);
		lin[1].push_back(clin[1]);
		lin[2].push_back(clin[2]);

		long long int ctime = com.GetLastSensorTimestamp();
		time.push_back(ctime);
	
		vector<double> cang = {(rot[0][i] - rot[0][i - 1])/(time[i] - time[i - 1]),
				     (rot[1][i] - rot[1][i - 1])/(time[i] - time[i - 1]),
				     (rot[2][i] - rot[2][i - 1])/(time[i] - time[i - 1])};
		ang[0].push_back(cang[0]);
		ang[1].push_back(cang[1]);
		ang[2].push_back(cang[2]);


		cout << (int)(((float)i / (float)d_points) * 100) << "% Complete" << '\r' << flush;
		this_thread::sleep_for(chrono::milliseconds(125));

	
	}

	cout << endl << endl;
	cout << "Data Collection Complete... Calculating Covariance..." << endl;
	//Calculate Covariances

	//vector<vector<double>> rot = {{1,2,3,4,5,6,6,7,8,6,3},{1,5,8,4,3,3,6,7,8,6,3},{1,2,3,4,5,6,7,8,6,3,4}};
	//vector<vector<double>> lin = {{1,2,3,4,5,6,6,7,8,6,3},{1,5,8,4,3,3,6,7,8,6,3},{1,2,3,4,5,6,7,8,6,3,4}};
	//vector<vector<double>> ang = {{1,2,3,4,5,6},{1,5,8,4,3,3},{1,2,3,4,5,6}};

	ofstream out;
	out.open("navx_calib.dat");

	cv::Mat rot_cov;
	cv::Mat rotin(rot.size(), rot[0].size(), CV_64FC1);
	for (size_t i = 0; i < rot.size(); i++) 
	{
		for (size_t j = 0; j < rot[0].size(); j++) {
		    rotin.at<double>(i, j) = rot[i][j];
		}
	}
    	cv::Mat rotout;
    	cv::Mat mean;
    	cv::calcCovarMatrix(rotin, rotout, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    	for(int i = 0; i < rotout.rows; i++)
    	{
        	const double* oi = rotout.ptr<double>(i);
        	for(int j = 0; j < rotout.cols; j++) {
            		if(isnan(oi[j]))
						out << 0 << endl;
					else
						out << oi[j] << endl;
			}
    	}
	out << endl;


	cv::Mat lin_cov;
	cv::Mat linin(lin.size(), lin[0].size(), CV_64FC1);
	for (size_t i = 0; i < lin.size(); i++) 
	{
		for (size_t j = 0; j < lin[0].size(); j++) {
		    linin.at<double>(i, j) = lin[i][j];
		}
	}
    	cv::Mat linout;
    	cv::calcCovarMatrix(linin, linout, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    	for(int i = 0; i < linout.rows; i++)
    	{
        	const double* oi = linout.ptr<double>(i);
        	for(int j = 0; j < linout.cols; j++) {
            		if(isnan(oi[j]))
						out << 0 << endl;
					else
						out << oi[j] << endl;
			}
    	}
	out << endl;


	cv::Mat ang_cov;
	cv::Mat angin(ang.size(), ang[0].size(), CV_64FC1);
	for (size_t i = 0; i < ang.size(); i++) 
	{
		for (size_t j = 0; j < ang[0].size(); j++) {
		    angin.at<double>(i, j) = ang[i][j];
		}
	}
    	cv::Mat angout;
    	cv::calcCovarMatrix(angin, angout, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    	for(int i = 0; i < angout.rows; i++)
    	{
        	const double* oi = angout.ptr<double>(i);
        	for(int j = 0; j < angout.cols; j++) {
            		if(isnan(oi[j]))
						out << 0 << endl;
					else
						out << oi[j] << endl;
			}
    	}
	out << endl;


	cv::Mat posein(lin.size() * 2, lin[0].size(), CV_64FC1);
	for (size_t i = 0; i < lin.size(); i++) 
	{
		for (size_t j = 0; j < lin[0].size(); j++) {
		    posein.at<double>(i, j) = lin[i][j];
		}
	}
	for (size_t i = 0; i < rot.size(); i++)
	{
		for(size_t j = 0; j < rot[0].size(); j++) {
		    posein.at<double>(i + lin.size(), j) = rot[i][j];
		}
	}
    	cv::Mat poseout;
    	cv::calcCovarMatrix(posein, poseout, mean, CV_COVAR_NORMAL | CV_COVAR_COLS);
    	for(int i = 0; i < poseout.rows; i++)
    	{
        	const double* oi = poseout.ptr<double>(i);
        	for(int j = 0; j < poseout.cols; j++)
            		out << oi[j] << endl;
    	}
	out << endl;


	out.close();

	cout << "Calibration complete, please copy navx_calib.dat to ROS wrapper source folder." << endl;

	return 0;
}


