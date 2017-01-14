#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "C920Camera.h"

using namespace cv;
using namespace std;
class BallDetector {

		Scalar min;
		Scalar max;
		int minArea;

		vector<vector<Point> > out;
		int area;
		Mat hsv;
		Mat mask;
		Mat opening;
		Mat closing;
		Mat res;
		Mat kernel;
		vector<vector<Point> > contours;

	public: BallDetector(Scalar mi, Scalar ma, int minAre=400) {
		min = mi;
		max = ma;
		minArea = minAre;
		kernel = Mat(5,5, CV_8U);
	}

	void changeMin(int hLo,int hUp,int sLo,int vLo) {
		min = Scalar(hLo,sLo,vLo);
		max.val[0] = hUp;
	}

	vector<vector<Point> > ballDetect(Mat frame)
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
				Rect rec = boundingRect(contours[x]);
				if (contourArea(contours[x]) > minArea && rec.width*rec.height < (frame.rows-10)*(frame.cols-10) ) {	//check to see the size of the contours
					out.push_back(contours[x]);	//add contours to output list
				}
			}
		}
		return out;
	}
};


float angleToDist(vector<Point> c) {
	if (c.size() == 0) throw invalid_argument("received empty contour");
	float a=.0010496;	//angle of each pixel .0010496 for c920 at 720p
	float b=1.54; //angle from top of view of camera to straight down in radians
	float h=.75;
	int sumOfY=0;
	for (int i = 0; i < c.size(); i++) {
		sumOfY+=c[i].y;
	}
	return h*tan(b-a*(sumOfY/c.size()));
}

float expectedSize(float dist) {
	float sizeConstant=5506;		//try 4669
	float h=.75;
	return sizeConstant/sqrt(pow(h,2)+pow(dist,2));

}

float ballCount(vector<Point> c) {
	if (c.size() == 0) throw invalid_argument("received empty contour");
	double dist = angleToDist(c);
	double eSize = expectedSize(dist);
	return contourArea(c)/eSize;
}

int main(int, char**)
{
	v4l2::C920Camera camera(1);
	camera.SetBrightness(60);
	camera.SetWhiteBalanceTemperature(30);
	camera.SetGain(45);
	camera.SetSaturation(190);
	camera.SetContrast(130);
	camera.ChangeCaptureSize(v4l2::CAPTURE_SIZE_1280x720);
	//VideoCapture cap(1); // open the default camera
    if(!camera.IsOpen())  // check if we succeeded
        return -1;


    namedWindow("frame",1);

	// create trackbars for color change
	// lower
	int hLo = 28;
	int sLo = 84;
	int vLo = 100;
	int hUp = 52;
	//int sUp = 255;
	//int vUp = 255;
	//int areaTrackbar = 10000;
	Scalar	min(28, 84, 100);
	Scalar	max(52, 255, 255);

	createTrackbar("HLo","frame",&hLo,179);
	createTrackbar("SLo","frame",&sLo,255);
	createTrackbar("VLo","frame",&vLo,255);
	createTrackbar("HUp","frame",&hUp,179);
	/*
	// upper
	createTrackbar("HUp","frame",&hUp,179);
	createTrackbar("SUp","frame",&sUp,255);
	createTrackbar("VUp","frame",&vUp,255);

	createTrackbar("areaTrackbar","frame",&areaTrackbar,50000);
*/
	double total;
	vector<vector<Point> > fuel;
	Mat frame;
	BallDetector b = BallDetector(min, max, 200);
	while (true) {
		camera.GrabFrame();
		camera.RetrieveMat(frame);
		//cap >> frame; // get a new frame from camera

		// get current positions of four trackbars
		hLo = getTrackbarPos("HLo","frame");
		sLo = getTrackbarPos("SLo","frame");
		vLo = getTrackbarPos("VLo","frame");
		hUp = getTrackbarPos("HUp","frame");
		/*
		sUp = getTrackbarPos("SUp","frame");
		vUp = getTrackbarPos("VUp","frame");

		areaTrackbar=getTrackbarPos("areaTrackbar","frame");

		Scalar	min(hLo, sLo, vLo);
	    Scalar	max(hUp, sUp, vUp);
*/
		b.changeMin(hLo,hUp,sLo,vLo);
		fuel = b.ballDetect(frame);
		total=0;
		drawContours(frame,fuel,-1,Scalar(255,0,0),2);
		for (int i = 0; i < fuel.size(); i++) {
			total += ballCount(fuel[i]);
			cout << "Dist " << i << ": " << angleToDist(fuel[i]) << " ";
		}
		cout <<"Fuel:" << total  << endl;
		imshow("frame", frame);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

