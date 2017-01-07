/*
 * Main.cpp
 *
 * Created on: Dec 31, 2014
 * Author: jrparks
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "camerain.cpp"

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    CameraIn vidIn(1, NULL);
    char name[PATH_MAX];
    int  index = 0;
    int  rc;
    struct stat statbuf;
    do
    {
        sprintf(name, "cap%d.avi", index++);
        rc = stat(name, &statbuf);
    } while (rc == 0);
    VideoWriter outputVideo(name, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(vidIn.width(), vidIn.height()), true);
    Mat frame;
	Mat depth;

    while (true)
    {
        if (vidIn.getFrame(frame, depth, false) && !frame.empty())
        {
            outputVideo << frame;
        }
        else
        {
            fprintf(stderr, "Unable to grab frame.\n");
            break;
        }
        imshow("Frame", frame);
        int wait_key = cv::waitKey(5);
        if ((wait_key == 27) || (wait_key == 32))
        {
            break;
        }
    }
    fprintf(stdout, "Closing camera.\n");
    return 0;
}
