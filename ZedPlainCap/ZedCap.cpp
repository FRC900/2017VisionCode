#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "zedcamerain.hpp"
#include "zmsin.hpp"
#include "zmsout.hpp"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	ZedIn *camera;
	if (argc == 2)
		camera = new ZMSIn(argv[1]);
	else
		camera = new ZedCameraIn(true);

	VideoWriter outputVideo;
	if (argc < 3)
	{
		char name[PATH_MAX];
		int  index = 0;
		int  rc;
		struct stat statbuf;
		do
		{
			sprintf(name, "camera%d.avi", index++);
			rc = stat(name, &statbuf);
		} while (rc == 0);
		outputVideo = VideoWriter(name, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(camera->width(), camera->height()), true);
	}
	Mat frame, depth;

	while (true)
	{
		if (camera->getFrame(frame, depth) && !frame.empty())
		{
			if (argc < 3)
			{
				outputVideo << frame;
			}
		}
		else
		{
			fprintf(stderr, "Unable to grab frame.\n");
			break;
		}
		if (argc < 3)
		{
			imshow("Frame", frame);
			uchar wait_key = cv::waitKey(2);
			if ((wait_key == 27) || (wait_key == 32))
			{
				break;
			}
		}
	}
	fprintf(stdout, "Closing camera.\n");
}
