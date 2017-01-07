#include <iostream>
#include <opencv2/opencv.hpp>
#include "FovisLocalizer.hpp"

using namespace std;
using namespace cv;

FovisLocalizer::FovisLocalizer(const CameraParams &input_params, const cv::Mat& initial_frame) :
	_rect(NULL),
	_odom(NULL)
{
	memset(&_rgb_params,0,sizeof(_rgb_params)); //set params to 0 to be sure

	_rgb_params.width = initial_frame.cols;
	_rgb_params.height = initial_frame.rows; //get width and height from the camera

	_rgb_params.fx = input_params.fx;
	_rgb_params.fy = input_params.fy;
	_rgb_params.cx = input_params.cx; 
	_rgb_params.cy = input_params.cy;

	_rect = new fovis::Rectification(_rgb_params);

	cvtColor(initial_frame, prevGray, CV_BGR2GRAY);

	reloadFovis();
}

void FovisLocalizer::reloadFovis()
{
	fovis::VisualOdometryOptions options = fovis::VisualOdometry::getDefaultOptions();
	options["max-pyramid-level"] = to_string(fv_param_max_pyr_level);
	options["feature-search-window"] = to_string(fv_param_feature_search_window);
	options["use-subpixel-refinement"] = "true";
	options["feature-window-size"] = to_string(fv_param_feature_window_size);
	options["target-pixels-per-feature"] = to_string(fv_param_target_ppf);

	if (_odom)
		delete _odom;
	_odom = new fovis::VisualOdometry(_rect, options);
}

FovisLocalizer::~FovisLocalizer()
{
	if (_rect)
		delete _rect;
	if (_odom)
		delete _odom;
}

void FovisLocalizer::processFrame(const cv::Mat& img_in, const cv::Mat& depth_in)
{
	int64 stepTimer;
	if (depth_in.empty())
		return;
	depthFrame = depth_in.clone();

	cvtColor(img_in, frameGray, CV_BGR2GRAY); // convert to grayscale 

	int num_optical_flow_sectors = num_optical_flow_sectors_x * num_optical_flow_sectors_y;

	stepTimer = cv::getTickCount();

	vector<Point2f> prevCorner, currCorner;
	vector< vector<Point2f> > prevCorner2(num_optical_flow_sectors); //holds an array of small optical point flow arrays
	vector< vector<Point2f> > currCorner2(num_optical_flow_sectors);
	vector< Rect > flow_rects;
	vector<uchar> status;
	vector<float> err;

	goodFeaturesToTrack(prevGray, prevCorner, num_optical_flow_points, 0.01, 30);
	calcOpticalFlowPyrLK(prevGray, frameGray, prevCorner, currCorner, status, err); //calculate optical flow
	prevGray = frameGray.clone();

	int flow_sector_size_x = img_in.cols / num_optical_flow_sectors_x;
	int flow_sector_size_y = img_in.rows / num_optical_flow_sectors_y;

	for(int i = 0; i < num_optical_flow_sectors_x; i++) {
		for(int j = 0; j < num_optical_flow_sectors_y; j++) {
			Rect sector_rect(Point(i * flow_sector_size_x,j * flow_sector_size_y), Point((i+1) * flow_sector_size_x,(j+1) * flow_sector_size_y));
			flow_rects.push_back(sector_rect);
		} //create rects to segment points found into boxes
	}

	// Status is set to true for each point where a match was found.
	// Use only these points for the rest of the calculations

	for(size_t i = 0; i < flow_rects.size(); i++) { //for each sector
		for(size_t j = 0; j < prevCorner.size(); j++) { //for each point
			if(status[j] && flow_rects[i].contains(prevCorner[j])) { //"contains" is a method to check if point is within a sector
				prevCorner2[i].push_back(prevCorner[j]); //add the point array to its repsective array
				currCorner2[i].push_back(currCorner[j]);
			}
		}
	}

	vector< float > optical_flow_magnitude;
	vector< bool > flow_good_sectors(prevCorner2.size(),true); //mark all sectors initially as good

	Mat rigid_transform;
	for(size_t i = 0; i < prevCorner2.size(); i++) { //for each sector calculate the magnitude of the movement
		if(prevCorner2[i].size() >= 3 && currCorner2[i].size() >= 3) 
		{ //don't calculate if no points in the sector
			rigid_transform = estimateRigidTransform(prevCorner2[i], currCorner2[i], false);
			if (rigid_transform.empty())
				flow_good_sectors[i] = false;
			cout << "Magnitude " << i << " : " << norm(rigid_transform,NORM_L2) << endl;
			optical_flow_magnitude.push_back(norm(rigid_transform,NORM_L2));
		}
		else
		{
			// Put dummy values here so flow_good_sectors indexes
			// line up with sector_rect indexes above and below
			flow_good_sectors[i] = false;
			optical_flow_magnitude.push_back(0.0);
		}
	}

	size_t num_sectors_left = flow_good_sectors.size();

	while(1) {
		float mag_mean;
		float sum = 0.;

		size_t good_sectors_prev_size = num_sectors_left;

		//calculate the mean flow magnitude of the remaining good sectors
		int num_good_sectors = 0;
		for(size_t i = 0; i < flow_good_sectors.size(); i++) {
			if (flow_good_sectors[i])
			{
				sum += optical_flow_magnitude[i]; 
				num_good_sectors += 1;
			}
		}
		if (num_good_sectors == 0)
			break;
		mag_mean = sum / (float)num_good_sectors;

		//this loop iterates through the points and checks if they 
		//are outside a range. if they are, then they are eliminated 
		//and the mean is recalculated
		for(size_t i = 0; i < flow_good_sectors.size(); i++) { 
			if(flow_good_sectors[i] && abs(optical_flow_magnitude[i]) > (flow_arbitrary_outlier_threshold_int / 100.0) * abs(mag_mean)) {
				cout << "Eliminating sector " << i << " because " <<  abs(optical_flow_magnitude[i]) << ">" << (flow_arbitrary_outlier_threshold_int / 100.0) * abs(mag_mean) << endl;
				flow_good_sectors[i] = false;
			}
		}

		num_sectors_left = 0;
		for(size_t i = 0; i < flow_good_sectors.size(); i++) //count number of sectors left
			if(flow_good_sectors[i]) 
				num_sectors_left++;

		if(good_sectors_prev_size == num_sectors_left) //if we failed to eliminate anything end this loop
			break;
	}
	float* ptr_depthFrame;

	for(size_t i = 0; i < flow_good_sectors.size(); i++) { //implement the optical flow into the depth data
		if(!flow_good_sectors[i]) { //true if the sector is bad
			Mat sector_submatrix = Mat(depthFrame,Range(flow_rects[i].tl().y,flow_rects[i].br().y), Range(flow_rects[i].tl().x,flow_rects[i].br().x));
			Mat(sector_submatrix.rows,sector_submatrix.cols,CV_32FC1,Scalar(-2.0)).copyTo(sector_submatrix); //copy
			//cout << "Sector " << i << " bad" << endl;
		} else {
		cout << "Sector " << i << " good" << endl;		
		}
	}

	for(int j = 0; j < depthFrame.rows; j++){ //for each row
		ptr_depthFrame = depthFrame.ptr<float>(j);
		for(int i = 0; i < depthFrame.cols; i++){ //for each pixel in row
			if(ptr_depthFrame[i] <= 0) {
				ptr_depthFrame[i] = NAN; //set to NaN if negative
			} else {
				ptr_depthFrame[i] /= 1000.0; //convert to m
			}
		}
	}

	fovis::DepthImage depthSource(_rgb_params, frameGray.cols, frameGray.rows);

	depthSource.setDepthImage((float*)depthFrame.data); //pass the data into fovis
	cout << "Time to optical flow - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;
	stepTimer = cv::getTickCount();
	uint8_t* pt = (uint8_t*)frameGray.data; //cast to unsigned integer and create pointer to data

	_odom->processFrame(pt, &depthSource); //run visual odometry

	Eigen::Isometry3d m = _odom->getMotionEstimate(); //estimate motion
	cout << "Time to fovis - " << ((double)cv::getTickCount() - stepTimer) / getTickFrequency() << endl;
	_transform_eigen = m;

	Eigen::Vector3d xyz = m.translation();
	Eigen::Vector3d rpy = m.rotation().eulerAngles(0, 1, 2);

	_transform.first[0] = xyz(0);
	_transform.first[1] = xyz(1);
	_transform.first[2] = xyz(2);

	_transform.second[0] = rpy(0);
	_transform.second[1] = rpy(1);
	_transform.second[2] = rpy(2);

	// If there were errors calculating the transformation
	// matrix, zero out everything so that this pose update
	// is ignored. 
	if (isnan(_transform.first[0]) ||
	    isnan(_transform.first[1]) ||
	    isnan(_transform.first[2]) ||
	    isnan(_transform.second[0]) ||
	    isnan(_transform.second[1]) ||
		isnan(_transform.second[2]) )
	{
		_transform.first[0] = 0;
		_transform.first[1] = 0;
		_transform.first[2] = 0;
		_transform.second[0] = 0;
		_transform.second[1] = 0;
		_transform.second[2] = 0;
	}


	//TODO : check that 2 180 rotations don't add up
	//to no rotation?
	for(int i = 0; i < 3; i++) {
		if(_transform.second[i] < -(M_PI/2.0) + (M_PI/4.0))
			while(_transform.second[i] <= -(M_PI/2.0) + (M_PI/4.0))
				_transform.second[i] += M_PI/2.0;
		else
			while(_transform.second[i] >= M_PI/2.0 - (M_PI/4.0))
				_transform.second[i] -= M_PI/2.0;
	}
	
	cout << "Camera X rotation: " << _transform.second[0] << endl;
	cout << "Camera Y rotation: " << _transform.second[1] << endl;
	cout << "Camera Z rotation: " << _transform.second[2] << endl;
}
