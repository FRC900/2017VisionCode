#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;
class BallDetector {

		Scalar min;
		Scalar max;
		int minArea;

		vector<Rect> out;
		int area;
		Mat hsv;
		Mat mask;
		Mat opening;
		Mat closing;
		Mat res;
		Mat kernel;
		vector<vector<Point> > contours;

	public: BallDetector(Scalar mi, Scalar ma, int minAre=2000) {
		min = mi;
		max = ma;
		minArea = minAre;
		kernel = Mat(5,5, CV_8U);
	}

	vector<Rect> ballDetect(Mat frame)
	{
		out.clear();	//reset the output list
		cvtColor(frame, hsv, CV_BGR2HSV);	//create an hsv image for filtering
		inRange(hsv, min, max, mask);	//create the mask
		bitwise_and(frame,frame, res, mask=mask);
		morphologyEx(mask, opening, MORPH_OPEN, kernel);	//expand the mask
		morphologyEx(opening, closing, MORPH_CLOSE, kernel);	//contract the mask
		//contours
		findContours(closing,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
		if (contours.size()>0) {	//check is there are any contours
			for (int x = 0; x < contours.size(); x++) {	//for each contours do stuff
				if (contourArea(contours[x]) > minArea) {	//check to see the size of the contours
					out.push_back(boundingRect(contours[x]));	//add contours to output list
				}
			}
		}
		return out;
	}
};

float sizeToDist(Rect r) {
	float k=.0001;
	float h=0;
	return 1/sqrt(k*pow((r.width+r.height)/2,2)-pow(h,2));

}

float angleToDist(Rect r) {
	float a=.001;	//angle of each pixel
	float b=1.4; //angle from top of view of camera to straight down in radians
	float h=1;
	cout << b-a*(r.y+r.height/2) << endl;
	return h*tan(b-a*(r.y+r.height/2));
}

int main(int, char**)
{
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("frame",1);

	// create trackbars for color change
	// lower
	//int hLo = 120;
	//int sLo = 72;
	//int vLo = 120;
	//int hUp = 179;
	//int sUp = 255;
	//int vUp = 255;
	//int areaTrackbar = 10000;
	Scalar	min(20, 80, 120);
	Scalar	max(50, 255, 255);
	/*
	createTrackbar("HLo","frame",&hLo,179);
	createTrackbar("SLo","frame",&sLo,255);
	createTrackbar("VLo","frame",&vLo,255);
	// upper
	createTrackbar("HUp","frame",&hUp,179);
	createTrackbar("SUp","frame",&sUp,255);
	createTrackbar("VUp","frame",&vUp,255);

	createTrackbar("areaTrackbar","frame",&areaTrackbar,50000);
*/

	vector<Rect> rects;
	Mat frame;
	BallDetector b = BallDetector(min, max, 2000);
	while (true) {
		cap >> frame; // get a new frame from camera
		/*
		// get current positions of four trackbars
		hLo = getTrackbarPos("HLo","frame");
		sLo = getTrackbarPos("SLo","frame");
		vLo = getTrackbarPos("VLo","frame");
		hUp = getTrackbarPos("HUp","frame");
		sUp = getTrackbarPos("SUp","frame");
		vUp = getTrackbarPos("VUp","frame");

		areaTrackbar=getTrackbarPos("areaTrackbar","frame");

		Scalar	min(hLo, sLo, vLo);
	    Scalar	max(hUp, sUp, vUp);
*/
		rects = b.ballDetect(frame);

		for (int i = 0; i < rects.size(); i++) {
			cout << angleToDist(rects[i]) << endl;
			rectangle(frame, rects[i], Scalar(0, 0, 255), 3);
		}
		imshow("frame", frame);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

