#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>
#include <vector>

#include "AHRS.h"

using namespace std;


void outer_product(vector<double> row, vector<double> col, vector<vector<double>>& dst);
void subtract(vector<double> row, double val, vector<double>& dst);
void add(vector<vector<double>> m, vector<vector<double>> m2, vector<vector<double>>& dst);
double mean(std::vector<double> &data);
void cov_matrix(vector<vector<double>> & d, vector<vector<double>> & dst);
void scale(vector<vector<double>> & d, double alpha);



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

	
	vector<vector<double>> rot_cov;
	cov_matrix(rot, rot_cov);

	vector<vector<double>> lin_cov;
	cov_matrix(lin, lin_cov);

	vector<vector<double>> ang_cov;
	cov_matrix(ang, ang_cov);

	//Export to File

	ofstream out;
	out.open("navx_calib.dat");

	for(unsigned int i = 0; i < rot_cov.size(); i++){ for(unsigned int j = 0; j < rot_cov[i].size(); j++){ out << rot_cov[i][j] << " "; }}
	out << endl;
	

	for(unsigned int i = 0; i < lin_cov.size(); i++){ for(unsigned int j = 0; j < lin_cov[i].size(); j++){ out << lin_cov[i][j] << " "; }}
	out << endl;


	for(unsigned int i = 0; i < ang_cov.size(); i++){ for(unsigned int j = 0; j < ang_cov[i].size(); j++){ out << ang_cov[i][j] << " "; }}

	out.close();

	cout << "Calibration complete, please copy navx_calib.dat to ROS wrapper source folder." << endl;

	return 0;
}

//Matrix Code Stolen From StackOverflow

void outer_product(vector<double> row, vector<double> col, vector<vector<double>>& dst) 
{
    for(unsigned i = 0; i < row.size(); i++) {
        for(unsigned j = 0; j < col.size(); i++) {
            dst[i][j] = row[i] * col[j];
        }
    }
}

void subtract(vector<double> row, double val, vector<double>& dst) 
{
    for(unsigned i = 0; i < row.size(); i++) {
        dst[i] = row[i] - val;
    }
}

void add(vector<vector<double>> m, vector<vector<double>> m2, vector<vector<double>>& dst) 
{
    for(unsigned i = 0; i < m.size(); i++) {
        for(unsigned j = 0; j < m[i].size(); j++) { 
            dst[i][j] = m[i][j] + m2[i][j];
        }
    }
}

double mean(std::vector<double> &data) 
{
    double mean = 0.0;

    for(unsigned i=0; (i < data.size());i++) {
        mean += data[i];
    }

    mean /= data.size();
    return mean;
}

void scale(vector<vector<double>> & d, double alpha) 
{
    for(unsigned i = 0; i < d.size(); i++) {
        for(unsigned j = 0; j < d[i].size(); j++) {
            d[i][j] *= alpha;
        }
    }
}

void cov_matrix(vector<vector<double>> & d, vector<vector<double>> & dst)
{
    for(unsigned i = 0; i < d.size(); i++) {
        double y_bar = mean(d[i]);
        vector<double> d_d_bar(d[i].size());
        subtract(d[i], y_bar, d_d_bar);
        vector<vector<double>> t(d.size());
        outer_product(d_d_bar, d_d_bar, t);
        add(dst, t, dst);
    } 
    scale(dst, 1/(d.size() - 1));
}

