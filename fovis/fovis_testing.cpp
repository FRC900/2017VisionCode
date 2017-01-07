#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#include <cstdio>
#include <ctime>

#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "persistence1d.hpp"

#include "FovisLocalizer.hpp"

#include "zedsvoin.hpp"
#include "zedcamerain.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

	MediaIn *cap = NULL;
	if(argc == 2) {
		cap = new ZedSVOIn(argv[1]);
		cerr << "Read SVO file" << endl;
	}
	else {
		cap = new ZedCameraIn();
		cerr << "Initialized camera" << endl;
	}

	Mat frame, depthFrame;

	clock_t startTime;

	cap->getFrame(frame, depthFrame);
	
	FovisLocalizer fvlc(cap->getCameraParams(), frame);

	while(1)
	{
		startTime = clock();

		cap->getFrame(frame, depthFrame);

		fvlc.processFrame(frame,depthFrame);

		imshow("frame",frame);

		cout << "XYZ " << fvlc.getTransform().first << endl;
		cout << "RPY " << fvlc.getTransform().second << endl;

		waitKey(5);
	}
}
