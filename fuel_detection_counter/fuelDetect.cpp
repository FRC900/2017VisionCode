#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "C920Camera.h"

using namespace cv;
using namespace std;
class FuelDetector {

		Scalar min;
		Scalar max;
		int minArea;
		float h;
		vector<vector<Point> > out;
		int area;
		Mat hsv;
		Mat mask;
		Mat opening;
		Mat closing;
		Mat res;
		Mat kernel;
		Mat output;
		vector<vector<Point> > contours;

	public: FuelDetector(float he) {
		max = Scalar(0,255,255);
		minArea=100;
		h=he;
		kernel = Mat(7,7, CV_8U);
	}

	void changeMin(int hLo,int hUp,int sLo,int vLo) {
		min = Scalar(hLo,sLo,vLo);
		max.val[0] = hUp;
	}

	vector<vector<Point> > getFuel(Mat frame)
	{
		out.clear();	//reset the output list
		cvtColor(frame, hsv, CV_BGR2HSV);	//create an hsv image for filtering
		inRange(hsv, min, max, mask);	//create the mask
		bitwise_and(frame,frame, res, mask=mask);
		morphologyEx(mask, opening, MORPH_OPEN, kernel);	//expand the mask
		morphologyEx(opening, closing, MORPH_CLOSE, kernel);	//contract the mask
		IplImage tmp=closing;
		cvFloodFill(&tmp,Point(0,0), Scalar(0));
		output = &tmp;
		imshow("output", output);
		//contours
		findContours(output,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
		if (contours.size()>0) {	//check is there are any contours
			for (int x = 0; x < contours.size(); x++) {
				Rect rec = boundingRect(contours[x]); //to implement screentoWorld want to pass this rectangle
				if (contourArea(contours[x]) > minArea && rec.width*rec.height < (frame.rows-60)*(frame.cols-60) ) {	//check to see the size of the contours
					out.push_back(contours[x]);	//add contours to output list Also Jude is the best.
				}
			}
		}
		return out;
	}

	Point3f angleToDist(vector<Point> c)
	{
		//X is left and right, Y is up and down, Z is foward and backward (away/towards camera)
		if (c.size() == 0) throw invalid_argument("received empty contour");
		Point2f fov (1.2290,.75572);	//angle of each pixel .0010496 for c920 at 720p
		float tFOV=1.57; //angle from top of view of camera to straight down in radians
		Point loc = Point(0,0);
		for (int i = 0; i < c.size(); i++)
		{
			loc.x+=c[i].x;
			loc.y+=c[i].y;
		}
		float z = h*tan(tFOV-fov.y/720*loc.y/c.size());

		float y = -h;
		float x = z * tan((fov.x/1280)*(loc.x/c.size())-(fov.x/2));
		return Point3f (x,y,z);
	}



	float expectedSize(Point3f d) {
		float sizeConstant=5220;		//try 4669
		return sizeConstant/sqrt(pow(d.x,2)+pow(d.y,2)+pow(d.z,2));

	}

	float fuelCount(vector<Point> c) {
		if (c.size() == 0) throw invalid_argument("received empty contour");
		Point3f location = angleToDist(c);
		double eSize = expectedSize(location);
		return contourArea(c)/eSize;
	}
	/* Found this online to possibly correct the fisheye distortion
	 * R_u = f*tan(theta) //projection by a pin whole camera, theta is the angle in rad between a point in the real world and the optical axis which goes from the center of the image through the center of the lens
	 * R_d = 2*f*sin(theta/2) //projection by common fisheye lens cameras (that is, distorted)
	 * R_u = f*tan(2*asin(R_d/(2*f))) //already know R_d and theta and if you knew the camera's focal length (represented by f) then correcting the image would amount to computing R_u in terms of R_d and theta.
	 * http://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
	 * http://wiki.panotools.org/Fisheye_Projection
	Point correct_fisheye(const Point& p,const Size& img)
	{
		// to polar
		const Point centre = {img.width/2,img.height/2};
		const Point rel = {p.x-centre.x,p.y-centre.y};
		const double theta = atan2(rel.y,rel.x);
		double R = sqrt((rel.x*rel.x)+(rel.y*rel.y));
		// fisheye undistortion in here please
		//... change R ...
		// back to rectangular
		const Point ret = Point(centre.x+R*cos(theta),centre.y+R*sin(theta));
		fprintf(stderr,"(%d,%d) in (%d,%d) = %f,%f = (%d,%d)\n",p.x,p.y,img.width,img.height,theta,R,ret.x,ret.y);
		return ret;
	}
	*/

};



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
	int hLo = 29;
	int sLo = 60;
	int vLo = 80;
	int hUp = 38;
	int k1 = 172781;
	int k2 = 550817;
	int p1 = 73542;
	int p2 = 89178;
	int k3 = 654275;

	//Mat cameraMatrix = Mat_<double>(3,3) << ( 9.5963091759762221e+02, 0., 6.7262624894813689e+02, 0., 9.7394188192143622e+02, 4.0219821536082321e+02, 0., 0., 1. );

	//createTrackbar("HLo","frame",&hLo,179);
	//createTrackbar("SLo","frame",&sLo,255);
	//createTrackbar("VLo","frame",&vLo,255);
	//createTrackbar("HUp","frame",&hUp,179);
	createTrackbar("k1","frame",&k1, 1000000);
	createTrackbar("k2","frame",&k2, 1000000);
	createTrackbar("p1","frame",&p1, 100000);
	createTrackbar("p2","frame",&p2, 100000);
	createTrackbar("k3","frame",&k3, 1000000);
	//double total;
	vector<vector<Point> > fuel;
	Mat oframe;
	Mat sframe;
	Mat frame;
	//camera matrix and distortion coefficent data is found in file c90_720p.yml file in current directory
	double cameraMatrixData[3][3] = {{9.5963091759762221e+02, 0., 6.7262624894813689e+02}, {0.,
       9.7394188192143622e+02, 4.0219821536082321e+02}, {0., 0., 1.}};
	double distCoeffsData[5];
	//double distCoeffsData[] = {1.7278156840527176e-01, 5.5081789263980618e-01, 7.3542979608589376e-03 , 8.9178305138385983e-03, 6.5427562481680634e-01};
	Mat cameraMatrix = Mat(3, 3, CV_64F, cameraMatrixData).inv();
	Mat distCoeffs = Mat(1, 5, CV_64F, distCoeffsData);
	FuelDetector b = FuelDetector(.70);
	b.changeMin(hLo,hUp,sLo,vLo);
	while (true)
	{
		camera.GrabFrame();
		camera.RetrieveMat(oframe);
		distCoeffsData[0] = -double(getTrackbarPos("k1","frame"))/1000000;
		distCoeffsData[1] = -double(getTrackbarPos("k2","frame"))/1000000;
		distCoeffsData[2] = double(getTrackbarPos("p1","frame"))/10000000;
		distCoeffsData[3] = double(getTrackbarPos("p2","frame"))/10000000;
		distCoeffsData[4] = double(getTrackbarPos("k3","frame"))/1000000;

		undistort(oframe,frame, cameraMatrix, distCoeffs); //undistort camera
		imshow("test3", frame);
		/*
		fuel = b.getFuel(frame);
		drawContours(frame,fuel,-1,Scalar(255,0,0),2);
		for (int i = 0; i < fuel.size(); i++)
		{
			cout << "Location: " << b.angleToDist(fuel[i]) << endl;
			cout << "Fuel Count: " << b.fuelCount(fuel[i]) <<endl;
		}
		imshow("frame", frame);
		*/
		if(waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

