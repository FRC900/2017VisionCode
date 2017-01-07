#include "FlowLocalizer.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

FlowLocalizer::FlowLocalizer(const cv::Mat &initial_frame)
{
	cvtColor(initial_frame, _prevFrame, CV_BGR2GRAY);
}

void FlowLocalizer::processFrame(const Mat &frame) 
{
	Mat currFrame;
	cvtColor(frame, currFrame, CV_BGR2GRAY);
	vector<Point2f> prevCorner, currCorner;
	vector<Point2f> prevCorner2, currCorner2;
	vector<uchar> status;
	vector<float> err;
	// Grab a set of features to track. Use optical flow to see
	// how how they move between frames.
	goodFeaturesToTrack(_prevFrame, prevCorner, 200, 0.01, 30);
	if (prevCorner.size())
	{
		calcOpticalFlowPyrLK(_prevFrame, currFrame, prevCorner, currCorner, status, err);

		// Status is set to true for each point where a match was found.
		// Use only these points for the rest of the calculations
		for (size_t i = 0; i < status.size(); i++)
		{
			if (status[i])
			{
				prevCorner2.push_back(prevCorner[i]);
				currCorner2.push_back(currCorner[i]);
			}
		}
	}

	// Return a transformation matrix which best maps 
	// points from prev to curr.
	// T = [ cos(angle) sin(angle) translation-x ]
	//     [-sin(angle) cos(angle) translation-y ] 
	Mat T;

	if (prevCorner2.size() && currCorner2.size())
		T = estimateRigidTransform(prevCorner2, currCorner2, false);

	// If a valid transformation is found, update predicted position
	// using it
	if (!T.empty())
	{
		// Pad the T matrix with a row (0, 0, 1)
		// to make it 3x3 - this makes the translation/rotation
		// to the next point a simple matrix multiply
		Mat Tpad(1,3,CV_64FC1);

		Tpad.at<double>(0,0) = 0.0;
		Tpad.at<double>(0,1) = 0.0;
		Tpad.at<double>(0,2) = 1.0;

		T.push_back(Tpad);
		
		//cout << "Optical Flow Transformation Matrix: " << T << endl;

		_transform_mat = T.clone();
	}
	else
	{
		_transform_mat = Mat::eye(3, 3, CV_64FC1);
	}

	//copy current frame to previous for next iteration
	_prevFrame = currFrame.clone();
}
