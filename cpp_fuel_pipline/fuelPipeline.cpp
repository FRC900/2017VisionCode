#include <iostream>
#include <vector>

using namespace std;
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::gpu;

int main(int, char**)
{

	//Initialize important info
	int hl=0, sl=0, vl=0, hu=0, su=0, vu=0, area_limit=0;
	int dp=0, min_dist=0, param1=0, param2=0, min_radius=0, max_radius=0;

	//Initialize important variables
	cv::Mat image, hsv, mask, mask2, hough;
	vector<vector<cv::Point> > contours, approx_contours;
	vector<cv::Point3f> fCircles;
	vector<cv::Point3i> iCircles;

	cv::resize(cv::imread("test.jpg"), image, cv::Size(cv::imread("test.jpg").cols * 0.45, cv::imread("test.jpg").rows * 0.455), 0.45, 0.455, cv::INTER_AREA);

	cv::Mat filler = cv::Mat::zeros(image.rows, image.cols, CV_8U);
	//GPU upload variables
	cv::gpu::GpuMat ghsv, /*gmask, */gmasks;//, gmask2, ghough, gBGR, gGRAY;
	cv::gpu::GpuMat kernal = cv::gpu::GpuMat(7, 7, CV_8U);

	//Create HSV filter Window
	cv::namedWindow("HSV");
	cv::createTrackbar("HL", "HSV", &hl, 180);
	cv::createTrackbar("SL", "HSV", &sl, 255);
	cv::createTrackbar("VL", "HSV", &vl, 255);
	cv::createTrackbar("HU", "HSV", &hu, 180);
	cv::createTrackbar("SU", "HSV", &su, 255);
	cv::createTrackbar("VU", "HSV", &vu, 255);
	cv::createTrackbar("AREA_LIMIT", "HSV", &area_limit, 1000);

	//Create Hough tracker Window
	cv::namedWindow("HOUGH");
	cv::createTrackbar("DP", "HOGUH", &dp, 20);
	cv::createTrackbar("MIN_DIST", "HOGUH", &min_dist, 100);
	cv::createTrackbar("PARAM1", "HOGUH", &param1, 700);
	cv::createTrackbar("PARAM2", "HOGUH", &param2, 200);
	cv::createTrackbar("MIN_RADIUS", "HOGUH", &min_radius, 100);
	cv::createTrackbar("MAX_RADIUS", "HOGUH", &max_radius, 100);
	while(true)
	{
		//Initialize imprtant variables
		/*hl = cv::getTrackbarPos("HL", "HSV");
		sl = cv::getTrackbarPos("SL", "HSV");
		vl = cv::getTrackbarPos("VL", "HSV");
		hu = cv::getTrackbarPos("HU", "HSV");
		su = cv::getTrackbarPos("SU", "HSV");
		vu = cv::getTrackbarPos("VU", "HSV");
		area_limit = cv::getTrackbarPos("AREA_LIMIT", "HSV");
		dp = cv::getTrackbarPos("DP", "HOUGH");
		min_dist = cv::getTrackbarPos("MIN_DIST", "HOUGH");
		param1 = cv::getTrackbarPos("PARAM1", "HOUGH");
		param2 = cv::getTrackbarPos("PARAM2", "HOUGH");
		min_radius = cv::getTrackbarPos("MIN_RADIUS", "HOUGH");
		max_radius = cv::getTrackbarPos("MAX_RADIUS", "HOUGH");
*/
		//GPU processing
		//inrange function
		cout<<typeid(image).name()<<endl;
		ghsv.upload(image);
		gmasks.upload(filler);
		/*cv::gpu::threshold(ghsv, gmasks, hl, hu, cv::THRESH_BINARY);
		cv::gpu::threshold(ghsv, gmasks, sl, su, cv::THRESH_BINARY);
		cv::gpu::threshold(ghsv, gmasks, vl, vu, cv::THRESH_BINARY);
		cv::gpu::bitwise_and(gmasks[0], gmasks[1], gmask);
		cv::gpu::bitwise_and(gmask, gmasks[2], gmask);

		//open mask
		gpu::morphologyEx(gmask, gmask, MORPH_OPEN, kernal);

		//push off of gpu
		gmask.download(mask);

		findContours(mask, contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < contours.length; i++)
		{
			//If contour area is too small skip it, but size doesn't matter
			if (contourArea(contours[i]) < area_limit)
				continue;

			epsilon = 0.005*arcLength(contours[i], true);
			approx = approxPolyDP(contours[i], epsilon, true);
			if (len(approx) < 20)
				continue;

			//append convex hull of the contours to fix ignoring the dimples... I like dimples.
			approx_contours.push_back(convexHull(contours[i]));
		}

		//Generate contour based mask
		mask2 = Mat(mask.shape, CV_8U);
		drawContours(mask2, approx_contours, -1, (255, 255, 255), -1);
		//imshow("CONTOUR_MASK", mask2);

		gmask2.uplaod(mask2);
		//Mask original image and create a grayscale image for hough circles
		gpu::bitwise_and(ghsv, ghsv, ghough, mask=gmask2); //Flip mask if necessary ()
		gpu::cvtColor(ghough, gBGR, CV_HSV2BGR);
		gpu::cvtColor(gBGR, gGRAY, CV_BGRVGRAY);

		//find hough circles
		//dp=3, mindDist=20, param1=500, param2=50, minRadius=5, maxRadius=20
		gpu::HoughCircles(gGRAY, fCircles, CV_HOUGH_GRADIENT, dp, minDist, param1, param2, minRadius, maxRadius);

		ghough.download(hough);

		if(!fCircles.empty())
		{
			//convert the (x, y) coordinates and radius of the circles to integers
			for(int i = 0; i < fCircles.size(); i++)
			{
				Point3i temp;
				temp = ((int)floor(fCircles[i].x), (int)floor(fCircles[i].y), (int)floor(fCircles[i].z));
				iCircles.push_back(temp);
			}
			//loop over the (x, y) coordinates and radius of the circles
			for(int i = 0; i < iCircles.size(); i++)
			{
				//draw the circle in the output image, then draw a rectangle
				//corresponding to the center of the circle
				circle(hough, (iCircles[i].x, iCircles[i].y), iCircles[i].z, (0, 255, 0), 4);
				rectangle(hough, (iCircles[i].x - 2, iCircles[i].y - 2), (iCircles[i].x + 2, iCircles[i].y + 2), (0, 128, 255), -1);
			}
		}
		drawContours(hsv, approx_contours, -1, (100, 255, 100), 2);
		imShow("HSV", hsv);
		imShow("Hough", hough);
*/
		int k = waitKey(10);
		if(k == 27)
		{
			if(!iCircles.empty())
			{
				cout << "Number of Circles: " <<  to_string(iCircles.size());
			}
			else
			{
				cout << "No Circles Found";
			}
		}
	}



}
