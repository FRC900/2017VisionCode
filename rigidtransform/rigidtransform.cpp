#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Simple class to track points on the screen
// Origin is where the tracking started
// Pos is where the code thinks the track should
// have moved to after the camera jumped around
class Tracker
{
	public :
		Point origin;
		Mat   pos;
		Tracker() :
			pos(Mat(3,1,CV_64FC1))
		{
		}
};

// Let the user click on the screen to pick
// an object to track.
void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		// Set the origin to the coords the user clicked on
		// Set the current position there too. Add the extra
		// row with a 1 in it to make the matrix math
		// work out
		((Tracker *)userdata)->origin              = Point(x,y);
		((Tracker *)userdata)->pos.at<double>(0,0) = x;
		((Tracker *)userdata)->pos.at<double>(1,0) = y;
		((Tracker *)userdata)->pos.at<double>(2,0) = 1.0;
	}
}

int main(int argc, char *argv[])
{
	// Take a video file as input
	if (argc < 1)
	{
		cout << argv[0] << " <video filename>" << endl;
		return -1;
	}

	VideoCapture cap(argv[1]);

	// Optical flow is done on grayscale images
	// Since there have to be 2 images to look
	// for differences, initialize one frame
	// here
	Mat prev, curr;
	Mat prevGray, currGray;
	cap >> prev;
	cvtColor(prev, prevGray, CV_BGR2GRAY);

	// Set up a mouse callback so the user can select 
	// where to start tracking from
	Tracker track;
	namedWindow("Frame");
	setMouseCallback("Frame", mouseCallback, &track);

	while (1)
	{
		cap >> curr;
		if (curr.empty())
			break;
		cvtColor(curr, currGray, CV_BGR2GRAY);

		vector<Point2f> prevCorner, currCorner;
		vector<Point2f> prevCorner2, currCorner2;
		vector<uchar> status;
		vector<float> err;
		// Grab a set of features to track. Use optical flow to see
		// how how they move between frames.
		goodFeaturesToTrack(prevGray, prevCorner, 200, 0.01, 30);
		calcOpticalFlowPyrLK(prevGray, currGray, prevCorner, currCorner, status, err);
		// Copy curr to prev for the next time around
		prevGray = currGray.clone();

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

		// Return a transformation matrix which best maps 
		// points from prev to curr.
		// T = [ cos(angle) sin(angle) translation-x ]
		//     [-sin(angle) cos(angle) translation-y ] 
		Mat T = estimateRigidTransform(prevCorner2, currCorner2, false);

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

			cout << T << endl;
			cout << track.pos << endl;

			// Matrix mult of T and current position gives
			// new position
			track.pos = T * track.pos;
		}

		Point target(track.pos.at<double>(0,0), track.pos.at<double>(1,0));
		line(curr, track.origin, target, Scalar(0, 255, 0), 3);
		circle(curr, target, 10, Scalar(0, 255, 0), -1);
		
		imshow ("Frame", curr);
		waitKey(0);
	}
	return 0;
}

