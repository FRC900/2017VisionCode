/*
 * Main.cpp
 *
 * Created on: Dec 31, 2014
 * Author: jrparks
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
#include "C920Camera.h"

using namespace std;
using namespace cv;

int g_h_max = 197;
int g_h_min = 102;
int g_s_max = 255;
int g_s_min = 72;
int g_v_max = 208;
int g_v_min = 0;
int Brightness = 128,
    Contrast = 128,
    Saturation = 128,
    Sharpness = 128,
    Gain = 20,
    Focus = 1,
    BacklightCompensation = 20,
    WhiteBalanceTemperature = 20,
	AutoExposure = 3;
v4l2::C920Camera camera(1);

void brightnessCallback(int value, void *data)
{
	Brightness = value;
	camera.SetBrightness(value);
}
void contrastCallback(int value, void *data)
{
	Contrast = value;
	camera.SetContrast(value);
}
void saturationCallback(int value, void *data)
{
	Saturation = value;
	camera.SetSaturation(value);
}
void sharpnessCallback(int value, void *data)
{
	Sharpness = value;
	camera.SetSharpness(value);
}
void gainCallback(int value, void *data)
{
	Gain = value;
	camera.SetGain(value);
}
void autoExposureCallback(int value, void *data)
{
	AutoExposure = value;
	camera.SetAutoExposure(value);
}
void backlightCompensationCallback(int value, void *data)
{
	BacklightCompensation = value;
	camera.SetBacklightCompensation(value);
}
void whiteBalanceTemperatureCallback(int value, void *data)
{
	WhiteBalanceTemperature = value;
	// Off by one to account for -1 being auto.
	camera.SetWhiteBalanceTemperature(value-1);
}
void focusCallback(int value, void *data)
{
	Focus = value;
	// Off by one to account for -1 being auto.
	camera.SetFocus(value-1);
}

int main(int argc, char* argv[]) {
	if (!camera.IsOpen()) {
		fprintf(stderr, "Unable to open camera.\n");
		return -1;
	}
	bool batch = false;

	if ((argc > 1) && (strcmp(argv[1], "--batch") == 0))
		batch = true;

	char name[PATH_MAX];
	int index = 0;
	int rc;
	struct stat statbuf;
	do
	{
		sprintf(name, "cap%d.avi", index++);
		rc = stat(name, &statbuf);
	}
	while (rc == 0);
	fprintf (stderr, "Writing to %s\n", name);

	camera.ChangeCaptureSize(v4l2::CAPTURE_SIZE_800x600);
	camera.ChangeCaptureFPS(v4l2::CAPTURE_FPS_30);
	camera.GetBrightness(Brightness);
	camera.GetContrast(Contrast);
	camera.GetSaturation(Saturation);
	camera.GetSharpness(Sharpness);
	camera.GetGain(Gain);
	camera.GetBacklightCompensation(BacklightCompensation);
	camera.GetWhiteBalanceTemperature(WhiteBalanceTemperature);
	++WhiteBalanceTemperature; // Off by 1 since -1 is auto

	// Set auto-focus off on startup
	focusCallback(1, NULL);
	autoExposureCallback(3, NULL);

	cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
	cv::createTrackbar("Brightness", "Adjustments", &Brightness, 255, brightnessCallback);
	cv::createTrackbar("Contrast", "Adjustments", &Contrast, 255, contrastCallback);
	cv::createTrackbar("Saturation", "Adjustments", &Saturation, 255, saturationCallback);
	cv::createTrackbar("Sharpness", "Adjustments", &Sharpness, 255, sharpnessCallback);
	cv::createTrackbar("Gain", "Adjustments", &Gain, 255, gainCallback);
	cv::createTrackbar("Backlight Compensation", "Adjustments", &BacklightCompensation, 1, backlightCompensationCallback);
	// Off by one to account for -1 being auto.
	cv::createTrackbar("White Balance Temperature", "Adjustments", &WhiteBalanceTemperature, 6501, whiteBalanceTemperatureCallback);
	cv::createTrackbar("Focus", "Adjustments", &Focus, 256, focusCallback);
	cv::createTrackbar("Auto Exposure", "Adjustments", &AutoExposure, 3, autoExposureCallback);

	cv::Mat frame;
	camera.GrabFrame();
	camera.RetrieveMat(frame);
	VideoWriter outputVideo(name, CV_FOURCC('M','J','P','G'), 30, Size(frame.cols, frame.rows), true);

	while (true) 
	{
		if (camera.GrabFrame() && camera.RetrieveMat(frame))
		{
			outputVideo << frame;
			if (!batch)
			{
				imshow( "Detect", frame);
				Mat btrack;
				inRange(frame, Scalar(g_h_min, g_s_min, g_v_min), Scalar(g_h_max, g_s_max, g_v_max), btrack);
				Mat element = getStructuringElement(MORPH_RECT, Size(5, 5), Point(2, 2));
				dilate(btrack, btrack, element);
				imshow("Tracking", btrack);
			}
		} 
		else 
		{
			fprintf(stderr, "Unable to grab frame from camera.\n");
			break;
		}
		if (!batch)
		{
			unsigned char wait_key = cv::waitKey(5);
			if (wait_key == 27 || wait_key == 32)
				break;
		}
	}
	fprintf(stdout, "Closing camera.\n");
	camera.Close();
}
