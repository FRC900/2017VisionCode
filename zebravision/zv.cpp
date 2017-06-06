#include <iostream>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>

#include <boost/filesystem.hpp>

#include "opencv2_3_shim.hpp"
#include "detectstate.hpp"
#include "frameticker.hpp"
#include "groundtruth.hpp"
#include "videoin.hpp"
#include "imagein.hpp"
#include "camerain.hpp"
#include "c920camerain.hpp"
#include "zedcamerain.hpp"
#include "zedsvoin.hpp"
#include "zmsin.hpp"
#include "aviout.hpp"
#include "pngout.hpp"
#include "zmsout.hpp"
#include "track3d.hpp"
#include "Args.hpp"
#include "WriteOnFrame.hpp"
#include "GoalDetector.hpp"
//#include "FovisLocalizer.hpp"
#include "Utilities.hpp"
#include "FlowLocalizer.hpp"
#include "ZvSettings.hpp"
#include "version.hpp"
#include "colormap.hpp"

using namespace std;
using namespace cv;
using namespace utils;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
using namespace cv::cuda;
#endif

//function prototypes
void writeImage(const Mat& frame, const vector<Rect>& rects, size_t index, const char *path, int frameNumber);
string getDateTimeString(void);
void drawRects(Mat image, const vector<Rect> &detectRects, Scalar rectColor = Scalar(0,0,255), bool text = true);
void drawTrackingInfo(Mat &frame, const vector<TrackedObjectDisplay> &displayList, const vector<vector<Point>> &posHist);
void drawTrackingTopDown(Mat &frame, const vector<TrackedObjectDisplay> &displayList);
bool openMedia(const string &readFileName, bool gui, const string &xmlFilename, MediaIn *&cap, string &capPath, string &windowName);
string getVideoOutName(bool raw, const char *suffix);

static bool isRunning = true;
static ZvSettings *zvSettings = NULL;

void my_handler(int s)
{
    if (s == SIGINT || s == SIGTERM)
    {
        isRunning = false;
    }
}

void drawRects(Mat image, const vector<Rect> &detectRects, Scalar rectColor, bool text)
{
    for (auto it = detectRects.cbegin(); it != detectRects.cend(); ++it)
    {
        // Mark detected rectangle on image
        // Change color based on direction we think the bin is pointing
        rectangle(image, *it, rectColor, 2);
        // Label each outlined image with a digit.  Top-level code allows
        // users to save these small images by hitting the key they're labeled with
        // This should be a quick way to grab lots of falsly detected images
        // which need to be added to the negative list for the next
        // pass of classifier training.
        if (text)
        {
            size_t i = it - detectRects.begin();
            if (i < 10)
            {
                stringstream label;
                label << i;
                putText(image, label.str(), Point(it->x + 10, it->y + 30), FONT_HERSHEY_PLAIN, 2.0, rectColor);
            }
        }
    }
}

const static double minRatio = 0.30;

void drawTrackingInfo(Mat& frame, const vector<TrackedObjectDisplay>& displayList, const vector<vector<Point>> &posHist)
{
    for (auto it = displayList.cbegin(); it != displayList.cend(); ++it)
    {
        if (it->ratio >= minRatio)
        {
            const int roundPosTo = 2;
			// Scaled ratio changes from [minRatio,1.0] -> [0,1] so we 
			// see the full range of colors 
			const double scaledRatio = (it->ratio - minRatio) / (1.0 - minRatio);
			const int scaledRatioInt = 255 * scaledRatio;
			//read from the array viridis defined in colormap.cpp
			const Scalar rectColor(viridis[scaledRatioInt * 3 + 2] * 255, viridis[scaledRatioInt * 3 + 1] * 255, viridis[scaledRatioInt * 3 + 0] * 255);
			//cout << scaledRatioInt << rectColor << endl;
			// Highlight detected target
			rectangle(frame, it->rect, rectColor, 3);
            // Write detect ID, distance and angle data
            putText(frame, it->id, Point(it->rect.x + 25, it->rect.y + 30), FONT_HERSHEY_PLAIN, 2.0, rectColor);
			putText(frame, it->name, Point(it->rect.x, it->rect.y + it->rect.height - 3), FONT_HERSHEY_PLAIN, 2.0, rectColor);
            stringstream label;
            label << fixed << setprecision(roundPosTo);
            label << "(" << it->position.x << "," << it->position.y << "," << it->position.z << ")";
            putText(frame, label.str(), Point(it->rect.x + 10, it->rect.y - 10), FONT_HERSHEY_PLAIN, 1.2, rectColor);

			if (posHist.size() > 0)
			{
				auto p = posHist.cbegin() + (it - displayList.cbegin());
				for (size_t i = 0; i < p->size(); i++)
				{
					const Point pt = (*p)[i];
					circle(frame, pt, 5, Scalar(0,192,192), -1);
					if (i > 0)
					{
						const Point prevPt = (*p)[i-1];
						line(frame, pt, prevPt, Scalar(0,192,192), 2);
					}
				}
			}
		}
	}
}


void drawTrackingTopDown(Mat& frame, const vector<TrackedObjectDisplay>& displayList, const Point3f& goalPos)
{
    //create a top view image of the robot and all detected objects
    Range xRange = Range(-4, 4);
    Range yRange = Range(-9, 9);

    Point imageSize   = Point(640, 640);
    Point imageCenter = Point(imageSize.x / 2, imageSize.y / 2);
    int   rectSize    = 40;

    frame = Mat(imageSize.y, imageSize.x, CV_8UC3, Scalar(0, 0, 0));
    circle(frame, imageCenter, 10, Scalar(0, 0, 255));
    line(frame, imageCenter, imageCenter - Point(0, imageSize.x / 2), Scalar(0, 0, 255), 3);
    for (auto it = displayList.cbegin(); it != displayList.cend(); ++it)
    {
        if (it->ratio >= minRatio)
        {
        Point2f realPos = Point2f(it->position.x, it->position.y);
        Point2f imagePos;
        imagePos.x = cvRound(realPos.x * (imageSize.x / (float)xRange.size()) + (imageSize.x / 2.0));
        imagePos.y = cvRound(-(realPos.y * (imageSize.y / (float)yRange.size())) + (imageSize.y / 2.0));
        circle(frame, imagePos, rectSize, Scalar(255, 0, 0), 5);
		}
    }
    if (goalPos != Point3f())
    {
        Point2f realPos = Point2f(goalPos.x, goalPos.y);
        Point2f imagePos;
        imagePos.x = cvRound(realPos.x * (imageSize.x / (float)xRange.size()) + (imageSize.x / 2.0));
        imagePos.y = cvRound(-(realPos.y * (imageSize.y / (float)yRange.size())) + (imageSize.y / 2.0));
        circle(frame, imagePos, rectSize, Scalar(255, 255, 0), 5);
    }
}

int main( int argc, const char** argv )
{
	// Flags for various UI features
	bool printFrames = false; // print frame number?
	bool gdDraw = false;      // draw goal detect details
	int frameDisplayFrequency = 1;

	// Read through command line args, extract
	// cmd line parameters and input filename
	Args args;

	//int64 stepTimer;

	cout << "Welcome to Zebravision v" << VERSION_MAJOR << "." << VERSION_MINOR << " " << GitDesc << endl;
	if (!args.processArgs(argc, argv))
		return -2;

	bool pause = !args.batchMode && args.pause;
	bool calibRects = false;
	bool filterUsingDepth = false;
	bool showTrackingHistory = false;

	//stuff to handle ctrl+c and escape gracefully
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = SA_SIGINFO;
	sigaction(SIGINT, &sigIntHandler, NULL);

	string windowName("Zebravision"); // GUI window name
	string capPath; // Output directory for captured images
	MediaIn* cap; //input object

	shared_ptr<tinyxml2::XMLDocument> capSettings;
	if (!openMedia(args.inputName, !args.batchMode, args.xmlFilename, cap, capPath, windowName))
	{
		cerr << "Could not open input file " << args.inputName << endl;
		return 0;
	}

	// Current frame data - BGR image and depth data (if available)
	Mat frame;
	Mat depth;

	// Seek to start frame if necessary
	if (args.frameStart > 0)
		cap->frameNumber(args.frameStart);

	//load an initial frame for stuff like optical flow which requires an initial
	// frame to compute difference against
	//also checks to make sure that the cap object works
	if (!cap->getFrame(frame, depth))
	{
		cerr << "Could not read frame from input " << args.inputName << endl;
		return 0;
	}

	GroundTruth groundTruth("ground_truth.txt", args.inputName);
	GroundTruth goalTruth("goal_truth.txt", args.inputName);

	//only start up the window if batch mode is disabled
	if (!args.batchMode)
		namedWindow(windowName, WINDOW_AUTOSIZE);

	Mat top_frame; // top-down view of tracked objects

	CameraParams camParams = cap->getCameraParams();

	//we need to detect from a maximum of 25 feet
	//use trigonometry to predict how big in pixels the object will be and set minDetectSize to that
	//float maxDistance = 25.0 * 12.0 * .0254; //ft * to_in * to_m
	//float angular_size = 2.0 * atan2(ObjectType(1).width(), (2.0*maxDistance));
	//minDetectSize = angular_size * (cap->width() / camParams.fov.x);
	minDetectSize = 60;
	cout << "Min Detect Size: " << minDetectSize << endl;
	d12Threshold = args.d12Threshold;
	d24Threshold = args.d24Threshold;
	c12Threshold = args.c12Threshold;
	c24Threshold = args.c24Threshold;
	// If UI is up, pop up the parameters window
	if (!args.batchMode && args.detection)
	{
		string detectWindowName = "Detection Parameters";
		namedWindow(detectWindowName);
		createTrackbar ("Scale", detectWindowName, &scale, 50);
		createTrackbar ("D12 NMS Threshold", detectWindowName, &d12NmsThreshold, 100);
		createTrackbar ("D24 NMS Threshold", detectWindowName, &d24NmsThreshold, 100);
		createTrackbar ("Min Detect", detectWindowName, &minDetectSize, 200);
		createTrackbar ("Max Detect", detectWindowName, &maxDetectSize, max(cap->width(), cap->height()));
		createTrackbar ("D12 Threshold", detectWindowName, &d12Threshold, 100);
		createTrackbar ("D24 Threshold", detectWindowName, &d24Threshold, 100);
		createTrackbar ("C12 Threshold", detectWindowName, &c12Threshold, 45);
		createTrackbar ("C24 Threshold", detectWindowName, &c24Threshold, 100);
	}

	// Create list of tracked objects
	TrackedObjectList objectTrackingList(Size(cap->width(),cap->height()), camParams.fov);

	FrameTicker frameTicker;

	//load up the neural networks.
	//Loads the highest epoch from the lowest numbered dir for each 
	//of the four net types.  By default this will be d12, d24, c12 and c24.
	//The code can switch to other nets if they're named like d12_1, d12_2, etc.
	DetectState *detectState = NULL;
	if (args.detection)
	{
		bool hasGPU = getCudaEnabledDeviceCount() > 0;
		detectState = new DetectState(
				ClassifierIO(args.d12BaseDir, args.d12DirNum, args.d12StageNum),
				ClassifierIO(args.d24BaseDir, args.d24DirNum, args.d24StageNum),
				ClassifierIO(args.c12BaseDir, args.c12DirNum, args.c12StageNum),
				ClassifierIO(args.c24BaseDir, args.c24DirNum, args.c24StageNum),
				camParams.fov.x, hasGPU, false, ObjectType(1));
	}

	// Find the first frame number which has ground truth data
	if (detectState && args.groundTruth)
	{
		int frameNum = groundTruth.nextFrameNumber();
		if (frameNum == -1)
			return 0;
		cap->frameNumber(frameNum);
	}

	// Open file to save raw video into. If depth data
	// is available, use a ZMS file since that can save
	// both RGB and depth info. If there's no depth
	// data write to and AVI instead.
	MediaOut *rawOut = NULL;
	if (args.writeVideo)
	{
		if (depth.empty())
			rawOut = new AVIOut(getVideoOutName(true, ".avi").c_str(), frame.size(), args.writeVideoSkip);
		else
			rawOut = new ZMSOut(getVideoOutName(true, ".zms").c_str(), args.writeVideoSkip, args.writeVideoSplit, args.writeVideoCompress);
	}

	// No point in saving ZMS files of processed output, since
	// the marked up frames will prevent us from re-running
	// the code through zv again.  This means we won't need
	// the depth info which is the only reason to use ZMS
	// files in the first place
	MediaOut *processedOut = NULL;
	if (args.saveVideo)
	{
		// Inputs with 1 frame are still images - write their
		// output as still images as well
		if (cap->frameCount() == 1)
			processedOut = new PNGOut(getVideoOutName(false, ".png").c_str());
		else
			processedOut = new AVIOut(getVideoOutName(false, ".avi").c_str(), frame.size(), args.saveVideoSkip);
	}

	//FovisLocalizer fvlc(cap->getCameraParams(), frame);

	//Creating optical flow computation object
	FlowLocalizer fllc(frame);

	//Creating Goaldetection object
	GoalDetector gd(camParams.fov, Size(cap->width(),cap->height()), !args.batchMode);

	// Start of the main loop
	//  -- grab a frame
	//  -- update the angle of tracked objects
	//  -- do a cascade detect on the current frame
	//  -- add those newly detected objects to the list of tracked objects
	while(isRunning)
	{
		frameTicker.mark(); // mark start of new frame

		// Write raw video before anything gets drawn on it
		if (rawOut)
			rawOut->saveFrame(frame, depth);

		// This code will load a classifier if none is loaded - this handles
		// initializing the classifier the first time through the loop.
		// It also handles cases where the user changes the classifer
		// being used - this forces a reload
		// Finally, it allows a switch between CPU and GPU on the fly
		if (detectState && (detectState->update() == false))
			break;

		// run Goaldetector
		gd.findBoilers(frame, depth);

		// Save detected contours to display later
		vector<vector<Point>> goal_contours;
		if (gdDraw)
			 goal_contours = gd.getContours(frame);

		if (gd.goal_pos() != Point3f())
			cout << "Goal Position=" << gd.goal_pos() << endl;

		// if we are using a goal_truth.txt file that has the actual 
		// locations of the goal mark that we detected correctly
        vector<Rect> goalTruthHit;
        vector<Rect> goalTruthMiss;
		if (cap->frameCount() >= 0)
		{
            vector<Rect> goalDetects;
			if (gd.goal_pos() != Point3f())
				goalDetects.push_back(gd.goal_rect());

			goalTruth.processFrame(cap->frameNumber(), goalDetects, 0.0001, goalTruthHit, goalTruthMiss);
			if (!args.batchMode && args.autoStop &&
				(goalTruthMiss.size() || (goalDetects.size() != goalTruthHit.size())))
				pause = true;
		}

		// compute optical flow
		if (detectState)
			fllc.processFrame(frame);

		// Apply the classifier to the frame
		// detectRects is a vector of rectangles, one for each detected object
		vector<Rect> detectRects;
		vector<Rect> uncalibDetectRects;
		if (detectState)
			detectState->detector()->Detect(frame, filterUsingDepth ? depth : Mat(), detectRects, uncalibDetectRects);

		// If args.captureAll is enabled, write each detected rectangle
		// to their own output image file. Do it before anything else
		// so there's nothing else drawn to frame yet, just the raw
		// input image
		if (args.captureAll)
			for (size_t index = 0; index < detectRects.size(); index++)
				writeImage(frame, detectRects, index, capPath.c_str(), cap->frameNumber());

		//adjust object locations based on optical flow information
		if (detectState)
		{
#if 0
			cout << "Locations before adjustment: " << endl;
			objectTrackingList.print();
#endif

			// TODO : switch this on and off based on depth info availability?
			//objectTrackingList.adjustLocation(fvlc.transform_eigen());
			objectTrackingList.adjustLocation(fllc.transform_mat());

#if 0
			cout << "Locations after adjustment: " << endl;
			objectTrackingList.print();
#endif
		}

		// Process detected rectangles - either match up with the nearest object
		// add it as a new one
		//stepTimer = cv::getTickCount();
		vector<Rect>depthFilteredDetectRects;
		vector<float> depths;
		vector<ObjectType> objTypes;
		const float depthRectScale = 0.2;
		for(auto it = detectRects.cbegin(); it != detectRects.cend(); ++it)
		{
			//cout << "Detected object at: " << *it;
			Rect depthRect = *it;

			//when we use optical flow to adjust we need to recompute the depth based on the new locations.
			//to do this shrink the bounding rectangle and take the minimum rect of the inside
			shrinkRect(depthRect,depthRectScale);
			Mat emptyMask(depth.rows,depth.cols,CV_8UC1,Scalar(255));
			float objectDepth = minOfDepthMat(depth, emptyMask, depthRect, 10).first;

			// If no depth data is available, calculate a fake
			// depth value as if the object were at the perfect
			// location for the detected rectangle size
			if (objectDepth < 0)
			{
				objectDepth = ObjectType(1).expectedDepth(*it, frame.size(), camParams.fov.x);
			}
			//cout << " Depth: " << objectDepth << endl;
			if (objectDepth > 0)
			{
				depthFilteredDetectRects.push_back(*it);
				depths.push_back(objectDepth);
				objTypes.push_back(ObjectType(1));
			}
		}

		objectTrackingList.processDetect(depthFilteredDetectRects, depths, objTypes);

		// Grab info from trackedobjects. Display it and update zmq subscribers
		vector<TrackedObjectDisplay> displayList;
        objectTrackingList.getDisplay(displayList);
		vector<vector<Point>> posHist;
		if (showTrackingHistory)
			posHist = objectTrackingList.getScreenPositionHistories();

		// Send data over the network
		// If objdetction is enabled, send detection data
		// always send goal detection info

		// Ground truth is a way of storing known locations of objects in a file.
		// Check ground truth data on videos and images,
		// but not on camera input
		vector<Rect> groundTruthHit;
		vector<Rect> groundTruthMiss;
		if (detectState && (cap->frameCount() >= 0))
			groundTruth.processFrame(cap->frameNumber(), detectRects, 0.45, groundTruthHit, groundTruthMiss);

		// For interactive mode, update the FPS as soon as we have
		// a complete array of frame time entries
		// For args.batch mode, only update every frameTicksLength frames to
		// avoid printing too much stuff
		if (frameTicker.valid() &&
			( (!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0)) ||
			  ( args.batchMode && (((cap->frameNumber() * ((args.skip > 0) ? args.skip : 1)) % 1) == 0))))
		{
			int frames = cap->frameCount();
			stringstream frameStr;
			frameStr << cap->frameNumber();
			if (frames > 0)
				frameStr << '/' << frames;

			stringstream fpsStr;
			float inFPS = cap->FPS();
			float outFPS = rawOut ? rawOut->FPS(): -1;
			if (inFPS > 0)
				fpsStr << fixed << setprecision(1) << inFPS << "G ";
			fpsStr << fixed << setprecision(2) << frameTicker.getFPS() << "M";
			if (outFPS > 0)
				fpsStr << " " << fixed << setprecision(1) << outFPS << "W";
			fpsStr << " FPS";
			if (!args.batchMode)
			{
				if (printFrames)
					putText(frame, frameStr.str(),
							Point(frame.cols - 15 * frameStr.str().length(), 45),
							FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
				putText(frame, fpsStr.str(), Point(frame.cols - 14 * fpsStr.str().length(), 20), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
			}
			else
				cerr << args.inputName << " : " << frameStr.str() << " : " << fpsStr.str() << endl;
		}

		// Various random display updates. Only do them every frameDisplayFrequency
		// frames. Normally this value is 1 so we display every frame. When exporting
		// X over a network, though, we can speed up processing by only displaying every
		// 3, 5 or whatever frames instead.
		if ((!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0)) ||
			(args.saveVideo && processedOut))
		{
			if (args.rects)
			{
				drawRects(frame, detectRects);
				if (calibRects)
					drawRects(frame, uncalibDetectRects, Scalar(0,255,255), false);
			}

			// Draw tracking info if it is enabled
			if (args.tracking)
				drawTrackingInfo(frame, displayList, posHist);

			// Actually display window if we're not in batch mode
			// Check is needed in case we're saving in batch mode
			// which means generate all the GUI data to write to 
			// disk but don't actually display on screen
			if (!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0)) 
			{
				vector<TrackedObjectDisplay> emptyDisplayList;
				drawTrackingTopDown(top_frame, args.tracking ? displayList : emptyDisplayList, gd.goal_pos());
				imshow("Top view", top_frame);
			}

			// Put an A on the screen if capture-all is enabled so
			// users can keep track of that toggle's mode
			if (args.captureAll)
				putText(frame, "A", Point(25,25), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255));
			if (filterUsingDepth)
				putText(frame, "D", Point(50,25), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255));

			// Display current classifier infomation
			if (detectState)
			{
				stringstream s;
				vector<size_t> debugInfo = detectState->detector()->DebugInfo();
				for (auto it = debugInfo.cbegin(); it != debugInfo.cend(); ++it)
				{
					s << *it;
					if ((it + 1) != debugInfo.cend())
						s << " / ";
				}
				putText(frame, s.str(),
						Point(0, frame.rows - 28), FONT_HERSHEY_PLAIN,
						1.5, Scalar(0,0,255));
				putText(frame, detectState->print(),
						Point(0, frame.rows - 8), FONT_HERSHEY_PLAIN,
						1.5, Scalar(0,0,255));
			}

			// Display crosshairs so we can line up the camera
			if (args.calibrate)
			{
			   line (frame, Point(frame.cols/2, 0) , Point(frame.cols/2, frame.rows), Scalar(255,255,0));
			   line (frame, Point(0, frame.rows/2) , Point(frame.cols, frame.rows/2), Scalar(255,255,0));
			   Rect gr = gd.goal_rect();
			   if (gr != Rect())
			   {

				   // Draw the ball to scale to our detected goal
				   int ballSize = cvRound(gr.width * 9.75 / 40.);
				   circle(frame, Point(frame.cols/2., gr.br().y - ballSize*2.), ballSize, Scalar(0,0,255), 3);
			   }
			}

			// Draw ground truth info for this frame. Will be a no-op
            // if none is available for this particular video frame
			if (args.rects)
			{
				if (detectState)
				{
					drawRects(frame, groundTruth.get(cap->frameNumber() - 1), Scalar(128, 0, 0), false);
					drawRects(frame, groundTruthHit, Scalar(128, 128, 128), false);
					drawRects(frame, groundTruthMiss, Scalar(0, 165, 255), false);
				}
				drawRects(frame, goalTruth.get(cap->frameNumber()), Scalar(0, 0, 128), false);
				drawRects(frame, goalTruthHit, Scalar(255, 0, 255), false);
				drawRects(frame, goalTruthMiss, Scalar(0, 165, 255), false);
			}

			//draw the goal along with debugging info if that's enabled
			if (gdDraw) 
				gd.drawOnFrame(frame, goal_contours);

			if (args.rects)
				rectangle(frame, gd.goal_rect(), Scalar(0, 255, 0));

			// Main call to display output for this frame after all
			// info has been written on it.
			if (!args.batchMode && ((cap->frameNumber() % frameDisplayFrequency) == 0)) 
				imshow(windowName, frame);

			// If saveVideo is set, write the marked-up frame to a file
			if (args.saveVideo && processedOut)
			{
				bool dateAndTime = true;
				WriteOnFrame textWriter;
				if (dateAndTime)
		 		{
					textWriter.writeTime(frame);
					textWriter.writeMatchNumTime(frame);
				} 
				// Make sure last frame finished 
				// writing, then write this one
				processedOut->sync();
				processedOut->saveFrame(frame, depth);
			}

			// Process user input for this frame
			char c = waitKey(5);
			if ((c == 'q') || (c == 27))
			{ // exit
				break;
			}
			else if( c == ' ')  // Toggle pause
			{
				pause = !pause;
			}
			else if( c == 'c')
			{
				calibRects = !calibRects;
			}
			else if( c == 'f')  // advance to next frame
			{
				pause = true;
				if (args.groundTruth)
				{
					int frame = groundTruth.nextFrameNumber();
					// Exit if no more frames left to test
					if (frame == -1)
						break;
					// Otherwise, if not paused, move to the next frame
					//TODO I don't think this will work as intended. Check it.
					cap->frameNumber(frame);
				}
				else if (args.skip > 1)
				{
					// Exit if the next skip puts the frame beyond the end of the video
					if ((cap->frameNumber() + args.skip) >= cap->frameCount())
						break;
					cap->frameNumber(cap->frameNumber() + args.skip);
				}

				// Force read of next frame
				// Subsequent calls to update/getFrame will return
				// same data since pause is set earlier
				// If either return false, that probably means EOF
				// so bail out
				if (!cap->getFrame(frame, depth, false))
					break;
			}
			else if (c == 'd')
			{
				filterUsingDepth = !filterUsingDepth;
			}
			else if (c == 'A') // toggle capture-all
			{
				args.captureAll = !args.captureAll;
			}
			else if (c == 't') // toggle args.tracking info display
			{
				args.tracking = !args.tracking;
			}
			else if (c == 'r') // toggle args.rects info display
			{
				args.rects = !args.rects;
			}
			else if (c == 'T') //toggle tagging of ground truth matching data
            {
                stringstream output_line;
                output_line << args.inputName << " " << cap->frameNumber() - 1 << " ";
                Rect r = gd.goal_rect();
                output_line << r.x << " " << r.y << " " << r.width << " " << r.height;
                ofstream tag_file("goal_truth.txt", std::ios_base::app | std::ios_base::out);
                tag_file << output_line.str() << endl;

                cout << "Tagged ground truth " << output_line.str() << endl;
            }
			else if (c == 'a') // save all detected images
			{
				// Save from a copy rather than the original
				// so all the markup isn't saved, only the raw image
				Mat frameCopy, depthCopy;
				if (cap->getFrame(frameCopy, depthCopy, true))
					for (size_t index = 0; index < detectRects.size(); index++)
						writeImage(frameCopy, detectRects, index, capPath.c_str(), cap->frameNumber());
				else
					break;
			}
			else if (c == 'p') // print frame number to console
			{
				cout << cap->frameNumber() << endl;
			}
			else if (c == 'P') // Toggle frame # printing to display
			{
				printFrames = !printFrames;
			}
			else if (c == 'S')
			{
				frameDisplayFrequency += 1;
			}
			else if (c == 's')
			{
				frameDisplayFrequency = max(1, frameDisplayFrequency - 1);
			}
			else if (c == 'g') // toggle Goal Detect drawing
			{
				gdDraw = !gdDraw;
			}
			else if (c == 'C') // toggle GIE mode
			{
				if (detectState)
					detectState->toggleCascade();
			}
			else if (c == '|')
			{
				Mat frameCopy, depthCopy;
				if(cap->getFrame(frameCopy, depthCopy))
				imwrite("savedImage.png",frameCopy);
			}
			else if (c == 'G') // toggle GPU/CPU detection mode
			{
				if (detectState)
					detectState->toggleGPU();
			}
			else if (c == 'h') // toggle tracking histories
			{
				showTrackingHistory = !showTrackingHistory;
			}
			else if (c == '.') // higher classifier stage
			{
				if (detectState)
				detectState->changeD12SubModel(true);
			}
			else if (c == ',') // lower classifier stage
			{
				if (detectState)
				detectState->changeD12SubModel(false);
			}
			else if (c == '>') // higher classifier dir num
			{
				if (detectState)
				detectState->changeD12Model(true);
			}
			else if (c == '<') // lower classifier dir num
			{
				if (detectState)
				detectState->changeD12Model(false);
			}
			else if (c == 'm') // higher classifier stage
			{
				if (detectState)
				detectState->changeD24SubModel(true);
			}
			else if (c == 'n') // lower classifier stage
			{
				if (detectState)
				detectState->changeD24SubModel(false);
			}
			else if (c == 'M') // higher classifier dir num
			{
				if (detectState)
				detectState->changeD24Model(true);
			}
			else if (c == 'N') // lower classifier dir num
			{
				if (detectState)
				detectState->changeD24Model(false);
			}
			else if (c == '\'') // higher classifier stage
			{
				if (detectState)
				detectState->changeC12SubModel(true);
			}
			else if (c == ';') // lower classifier stage
			{
				if (detectState)
				detectState->changeC12SubModel(false);
			}
			else if (c == '\"') // higher classifier dir num
			{
				if (detectState)
				detectState->changeC12Model(true);
			}
			else if (c == ':') // lower classifier dir num
			{
				if (detectState)
				detectState->changeC12Model(false);
			}
			else if (c == 'l') // higher classifier stage
			{
				if (detectState)
				detectState->changeC24SubModel(true);
			}
			else if (c == 'k') // lower classifier stage
			{
				if (detectState)
				detectState->changeC24SubModel(false);
			}
			else if (c == 'L') // higher classifier dir num
			{
				if (detectState)
				detectState->changeC24Model(true);
			}
			else if (c == 'K') // lower classifier dir num
			{
				if (detectState)
				detectState->changeC24Model(false);
			}
			
			else if (isdigit(c)) // save a single detected image
			{
				Mat frameCopy, depthCopy;
				if (cap->getFrame(frameCopy, depthCopy, true))
					writeImage(frameCopy, detectRects, c - '0', capPath.c_str(), cap->frameNumber());
				else 
					break;
			}
		}

		// If testing only ground truth frames, move to the
		// next one in the list
		if (detectState && args.groundTruth)
		{
			int frame = groundTruth.nextFrameNumber();
			// Exit if no more frames left to test
			if (frame == -1)
				break;
			// Otherwise, if not paused, move to the next frame
			if (!pause)
				cap->frameNumber(frame);
		}
		// Skip over frames if needed - useful for batch extracting hard negatives
		// so we don't get negatives from every frame. Sequential frames will be
		// pretty similar so there will be lots of redundant images found
		else if (!pause && (args.skip > 0))
		{
			// Exit if the next skip puts the frame beyond the end of the video
			if ((cap->frameNumber() + args.skip) >= cap->frameCount())
				break;
			cap->frameNumber(cap->frameNumber() + args.skip);
		}

		// Check for running still images in batch mode - only
		// process the image once rather than looping forever
		if (args.batchMode && (cap->frameCount() == 1))
			break;

		if (!cap->getFrame(frame, depth, pause))
			break;
	}

	if (detectState)
	{
		cout << "Ball detect ground truth : " << endl;
		groundTruth.print();
	}
	cout << endl << "Goal detect ground truth : " << endl;
	goalTruth.print();

	if (detectState)
		delete detectState;
  	if(cap)
    	delete cap;
	return 0;
}


// Write out the selected rectangle from the input frame
void writeImage(const Mat& frame, const vector<Rect>& rects, size_t index, const char *path, int frameNumber)
{
    mkdir("negative", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (index < rects.size())
    {
        // Create filename, save image
        stringstream fn;
        fn << "negative/" << path << "_" << frameNumber << "_" << index;
        imwrite(fn.str() + ".png", frame(rects[index]));
    }
}


string getDateTimeString(void)
{
    time_t    rawtime;
    struct tm *timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    stringstream ss;
	ss << setfill('0') << setw(4) << timeinfo->tm_year + 1900 << "-" << setw(2) << timeinfo->tm_mon + 1 << "-" << timeinfo->tm_mday << "_";
	ss << setfill('0') << setw(2) << timeinfo->tm_hour << "-" << setw(2) << timeinfo->tm_min << "-" << setw(2) << timeinfo->tm_sec;
    return ss.str();
}


// Open video capture object. Figure out if input is camera, video, image, etc
bool openMedia(const string &readFileName, bool gui, const string &xmlFilename, MediaIn *&cap, string &capPath, string &windowName)
{
	zvSettings = new ZvSettings(xmlFilename);

	// Digit, but no dot (meaning no file extension)? Open camera
	if (readFileName.length() == 0 ||
		((readFileName.find('.') == string::npos) && isdigit(readFileName[0])))
	{
		stringstream ss;
		int camera = readFileName.length() ? atoi(readFileName.c_str()) : 0;

		cap = new ZedCameraIn(gui, zvSettings);
		if (!cap->isOpened())
		{
			delete cap;
			cap = new C920CameraIn(camera, gui, zvSettings);
			if (!cap->isOpened())
			{
				delete cap;
				cap = new CameraIn(camera, zvSettings);
				ss << "Default Camera ";
			}
			else
			{
				ss << "C920 Camera ";
			}
		}
		else
		{
			ss << "Zed Camera ";
		}
		ss << camera;
		windowName = ss.str();
		capPath    = getDateTimeString();
	}
	else // has to be a file name, we hope
	{
		string ext = boost::filesystem::extension(readFileName);
		if ((ext == ".png") || (ext == ".jpg") ||
		    (ext == ".PNG") || (ext == ".JPG"))
			cap = new ImageIn((char*)readFileName.c_str(), zvSettings);
		else if ((ext ==  ".svo") || (ext ==  ".SVO"))
			cap = new ZedSVOIn(readFileName.c_str(), zvSettings);
		else if ((ext == ".zms") || (ext == ".ZMS"))
			cap = new ZMSIn(readFileName.c_str(), zvSettings);
		else
			cap = new VideoIn(readFileName.c_str(), zvSettings);

		// Strip off directory for capture path
		capPath = readFileName;
		const size_t last_slash_idx = capPath.find_last_of("\\/");
		if (std::string::npos != last_slash_idx)
			capPath.erase(0, last_slash_idx + 1);
		windowName = readFileName;
	}
	return true;
}

// Video-YYYY-MM-DD_hr-min-sec-##.avi
string getVideoOutName(bool raw, const char *suffix)
{
    int          index = 0;
    int          rc;
    stringstream ss;

    do
    {
        ss.str(string(""));
        ss.clear();
        ss << "Video-" << getDateTimeString() << "-";
        ss << index++;
        if (raw == false)
        {
            ss << "_processed";
        }
		if (suffix)
		{
			ss << suffix;
		}
		struct stat  statbuf;
        rc = stat(ss.str().c_str(), &statbuf);
    }
	while (rc == 0);
    return ss.str();
}
