#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>
#include "Args.hpp"

using namespace std;

static void Usage(void)
{
   cout << "Usage : test [option list] [camera # | input file name]" << endl << endl;
   cout << "  Camera number is an integer corresponding to /dev/video??" << endl;
   cout << "  Input files can be video, combined video & depth (ZMS or SVO) or PNG/JPG images" << endl;

   cout << "\t--frame=<frame num>  start at given frame" << endl;
   cout << "\t--all                write to disk all detected images in each frame" << endl;
   cout << "\t--batch              run without GUI" << endl;
   cout << "\t--skip=<x>           skip frames, only processing every <x> - file input only" << endl;
   cout << "\t--pause              start paused" << endl;
   cout << "\t--calibrate          bring up crosshair to calibrate camera position" << endl;
   cout << "\t--capture            write raw camera video to output file" << endl;
   cout << "\t--captureSkip=       only write one of every N frames" << endl;
   cout << "\t--save               write processed video to output file" << endl;
   cout << "\t--saveSkip=          only write one of every N processed frames" << endl;
   cout << "\t--no-rects           start with detection rectangles disabled" << endl;
   cout << "\t--no-tracking        start with tracking rectangles disabled" << endl;
   cout << "\t--no-detection       disable object detection" << endl;
   cout << "\t--d12Base=           base directory for d12 info" << endl;
   cout << "\t--d12Dir=            pick d12 dir and stage number" << endl;
   cout << "\t--d12Stage=          from command line" << endl;
   cout << "\t--d12Threshold=      set d12 detection threshold" << endl;
   cout << "\t--d24Base=           base directory for d24 info" << endl;
   cout << "\t--d24Dir=            pick d24 dir and stage number" << endl;
   cout << "\t--d24Stage=          from command line" << endl;
   cout << "\t--d24Threshold=      set d24 detection threshold" << endl;
   cout << "\t--c12Base=           base directory for c12 info" << endl;
   cout << "\t--c12Dir=            pick c12 dir and stage number" << endl;
   cout << "\t--c12Stage=          from command line" << endl;
   cout << "\t--c12Threshold=      set c12 detection threshold" << endl;
   cout << "\t--c24Base=           base directory for c24 info" << endl;
   cout << "\t--c24Dir=            pick c24 dir and stage number" << endl;
   cout << "\t--c24Stage=          from command line" << endl;
   cout << "\t--c24Threshold=      set c24 detection threshold" << endl;
   cout << "\t--groundTruth        only test frames which have ground truth data " << endl;
   cout << "\t--xmlFile=           XML file to read/write settings to/from" << endl;
   cout << endl;
   cout << "Examples:" << endl;
   cout << "test : start in GUI mode, open default camera, start detecting and tracking while displaying results in the GUI" << endl;
   cout << "test --batch --capture : Start without a gui, write captured video to disk.  This is the normal way the code is run on the robot during matches. The code communicates via ZMQ to the roborio." << endl;
   cout << "test --batch --all foo.mp4 : run through foo.mp4. For each frame, write to disk images of all recognized objects. Might be useful for grabbing negative samples from a video known to have no positives in it" << endl;
}

Args::Args(void)
{
	captureAll         = false;
	tracking           = true;
	rects              = true;
	batchMode          = false;
	pause              = false;
	skip               = 0;
	calibrate          = false;
	writeVideo         = false;
	writeVideoSkip     = 1;
	saveVideo          = false;
	saveVideoSkip      = 1;
	detection          = true;
	d12BaseDir         = "/home/ubuntu/2016VisionCode/zebravision/d12";
	d12DirNum          = -1;
	d12StageNum        = -1;
	d12Threshold       = 45;
	d24BaseDir         = "/home/ubuntu/2016VisionCode/zebravision/d24";
	d24DirNum          = -1;
	d24StageNum        = -1;
	d24Threshold       = 98;
	c12BaseDir         = "/home/ubuntu/2016VisionCode/zebravision/c12";
	c12DirNum          = -1;
	c12StageNum        = -1;
	c12Threshold       = 21;
	c24BaseDir         = "/home/ubuntu/2016VisionCode/zebravision/c24";
	c24DirNum          = -1;
	c24StageNum        = -1;
	c24Threshold       = 21;
	frameStart         = 0.0;
	groundTruth        = false;
	xmlFilename        = "/home/ubuntu/2016VisionCode/zebravision/settings.xml";
}

bool Args::processArgs(int argc, const char **argv)
{
	const string frameOpt           = "--frame=";          // start at given frame
	const string captureAllOpt      = "--all";             // capture all detected images in each frame
	const string batchModeOpt       = "--batch";           // run without GUI
	const string pauseOpt           = "--pause";           // start paused
	const string skipOpt            = "--skip=";           // skip frames in input video file
	const string calibrateOpt       = "--calibrate";       // bring up crosshair to calibrate camera position
	const string writeVideoOpt      = "--capture";         // save camera video to output file
	const string writeVideoSkipOpt  = "--captureSkip=";    // skip frames in output video file
	const string saveVideoOpt       = "--save";            // write processed video to output file
	const string saveVideoSkipOpt   = "--saveSkip=";       // only write every N frames of processed video
	const string rectsOpt           = "--no-rects";        // start with detection rectangles disabled
	const string trackingOpt        = "--no-tracking";     // start with tracking rectangles disabled
	const string detectOpt          = "--no-detection";    // disable object detection
	const string d12BaseOpt         = "--d12Base=";        // d12 base dir
	const string d12DirOpt          = "--d12Dir=";         // pick d12 dir and stage number
	const string d12StageOpt        = "--d12Stage=";       // from command line
	const string d12ThresholdOpt    = "--d12Threshold=";    
	const string d24BaseOpt         = "--d24Base=";        // d24 base dir
	const string d24DirOpt          = "--d24Dir=";         // pick d24 dir and stage number
	const string d24StageOpt        = "--d24Stage=";       // from command line
	const string d24ThresholdOpt    = "--d24Threshold=";    
	const string c12BaseOpt         = "--c12Base=";        // c12 base dir
	const string c12DirOpt          = "--c12Dir=";         // pick c12 dir and stage number
	const string c12StageOpt        = "--c12Stage=";       // from command line
	const string c12ThresholdOpt    = "--c12Threshold=";    
	const string c24BaseOpt         = "--c24Base=";        // c24 base dir
	const string c24DirOpt          = "--c24Dir=";         // pick c24 dir and stage number
	const string c24StageOpt        = "--c24Stage=";       // from command line
	const string c24ThresholdOpt    = "--c24Threshold=";    
	const string groundTruthOpt     = "--groundTruth";     // only test frames which have ground truth data
	const string xmlFileOpt         = "--xmlFile=";        // read camera settings from XML file
	const string badOpt             = "--";
	// Read through command line args, extract
	// cmd line parameters and input filename
	int fileArgc;
	for (fileArgc = 1; fileArgc < argc; fileArgc++)
	{
		if (frameOpt.compare(0, frameOpt.length(), argv[fileArgc], frameOpt.length()) == 0)
			frameStart = atoi(argv[fileArgc] + frameOpt.length());
		else if (captureAllOpt.compare(0, captureAllOpt.length(), argv[fileArgc], captureAllOpt.length()) == 0)
			captureAll = true;
		else if (batchModeOpt.compare(0, batchModeOpt.length(), argv[fileArgc], batchModeOpt.length()) == 0)
			batchMode = true;
		else if (pauseOpt.compare(0, pauseOpt.length(), argv[fileArgc], pauseOpt.length()) == 0)
			pause = true;
		else if (skipOpt.compare(0, skipOpt.length(), argv[fileArgc], skipOpt.length()) == 0)
			skip = atoi(argv[fileArgc] + skipOpt.length());
		else if (calibrateOpt.compare(0, calibrateOpt.length(), argv[fileArgc], calibrateOpt.length()) == 0)
			calibrate = true;
		else if (writeVideoSkipOpt.compare(0, writeVideoSkipOpt.length(), argv[fileArgc], writeVideoSkipOpt.length()) == 0)
			writeVideoSkip = atoi(argv[fileArgc] + writeVideoSkipOpt.length());
		else if (writeVideoOpt.compare(0, writeVideoOpt.length(), argv[fileArgc], writeVideoOpt.length()) == 0)
			writeVideo = true;
		else if (saveVideoSkipOpt.compare(0, saveVideoSkipOpt.length(), argv[fileArgc], saveVideoSkipOpt.length()) == 0)
			saveVideoSkip = atoi(argv[fileArgc] + saveVideoSkipOpt.length());
		else if (saveVideoOpt.compare(0, saveVideoOpt.length(), argv[fileArgc], saveVideoOpt.length()) == 0)
			saveVideo = true;
		else if (detectOpt.compare(0, detectOpt.length(), argv[fileArgc], detectOpt.length()) == 0)
			detection = false;
		else if (trackingOpt.compare(0, trackingOpt.length(), argv[fileArgc], trackingOpt.length()) == 0)
			tracking = false;
		else if (rectsOpt.compare(0, rectsOpt.length(), argv[fileArgc], rectsOpt.length()) == 0)
			rects = false;
		else if (d12BaseOpt.compare(0, d12BaseOpt.length(), argv[fileArgc], d12BaseOpt.length()) == 0)
			d12BaseDir = string(argv[fileArgc] + d12BaseOpt.length());
		else if (d12DirOpt.compare(0, d12DirOpt.length(), argv[fileArgc], d12DirOpt.length()) == 0)
			d12DirNum = atoi(argv[fileArgc] + d12DirOpt.length());
		else if (d12StageOpt.compare(0, d12StageOpt.length(), argv[fileArgc], d12StageOpt.length()) == 0)
			d12StageNum = atoi(argv[fileArgc] + d12StageOpt.length());
		else if (d12ThresholdOpt.compare(0, d12ThresholdOpt.length(), argv[fileArgc], d12ThresholdOpt.length()) == 0)
			d12Threshold = atoi(argv[fileArgc] + d12ThresholdOpt.length());
		else if (d24BaseOpt.compare(0, d24BaseOpt.length(), argv[fileArgc], d24BaseOpt.length()) == 0)
			d24BaseDir = string(argv[fileArgc] + d24BaseOpt.length());
		else if (d24DirOpt.compare(0, d24DirOpt.length(), argv[fileArgc], d24DirOpt.length()) == 0)
			d24DirNum = atoi(argv[fileArgc] + d24DirOpt.length());
		else if (d24StageOpt.compare(0, d24StageOpt.length(), argv[fileArgc], d24StageOpt.length()) == 0)
			d24StageNum = atoi(argv[fileArgc] + d24StageOpt.length());
		else if (d24ThresholdOpt.compare(0, d24ThresholdOpt.length(), argv[fileArgc], d24ThresholdOpt.length()) == 0)
			d24Threshold = atoi(argv[fileArgc] + d24ThresholdOpt.length());
		else if (c12BaseOpt.compare(0, c12BaseOpt.length(), argv[fileArgc], c12BaseOpt.length()) == 0)
			c12BaseDir = string(argv[fileArgc] + c12BaseOpt.length());
		else if (c12DirOpt.compare(0, c12DirOpt.length(), argv[fileArgc], c12DirOpt.length()) == 0)
			c12DirNum = atoi(argv[fileArgc] + c12DirOpt.length());
		else if (c12StageOpt.compare(0, c12StageOpt.length(), argv[fileArgc], c12StageOpt.length()) == 0)
			c12StageNum = atoi(argv[fileArgc] + c12StageOpt.length());
		else if (c12ThresholdOpt.compare(0, c12ThresholdOpt.length(), argv[fileArgc], c12ThresholdOpt.length()) == 0)
			c12Threshold = atoi(argv[fileArgc] + c12ThresholdOpt.length());
		else if (c24BaseOpt.compare(0, c24BaseOpt.length(), argv[fileArgc], c24BaseOpt.length()) == 0)
			c24BaseDir = string(argv[fileArgc] + c24BaseOpt.length());
		else if (c24DirOpt.compare(0, c24DirOpt.length(), argv[fileArgc], c24DirOpt.length()) == 0)
			c24DirNum = atoi(argv[fileArgc] + c24DirOpt.length());
		else if (c24StageOpt.compare(0, c24StageOpt.length(), argv[fileArgc], c24StageOpt.length()) == 0)
			c24StageNum = atoi(argv[fileArgc] + c24StageOpt.length());
		else if (c24ThresholdOpt.compare(0, c24ThresholdOpt.length(), argv[fileArgc], c24ThresholdOpt.length()) == 0)
			c24Threshold = atoi(argv[fileArgc] + c24ThresholdOpt.length());
		else if (groundTruthOpt.compare(0, groundTruthOpt.length(), argv[fileArgc], groundTruthOpt.length()) == 0)
			groundTruth = true;
		else if (xmlFileOpt.compare(0, xmlFileOpt.length(), argv[fileArgc], xmlFileOpt.length()) == 0)
			xmlFilename = string(argv[fileArgc] + xmlFileOpt.length());
		else if (badOpt.compare(0, badOpt.length(), argv[fileArgc], badOpt.length()) == 0) // unknown option
		{
			cerr << "Unknown command line option " << argv[fileArgc] << endl;
			Usage();
			return false;
		}
		else // first non -- arg is filename or camera number
			break;
	}
	if (argc > (fileArgc + 1))
	{
	   cerr << "Extra arguments after file name" << endl;
	   Usage();
	   return false;
	}
	if (fileArgc < argc)
		inputName = argv[fileArgc];
	return true;
}
