// Tool to create goal_truth.txt file. This file holds information on the
// locations of goals in an input video file.  The main zv executable can 
// use this file to automatically test how well goal detection is working
// The input file is a ZMS video - this includes RGB plus depth data
// The user highlights the location of a goal in a frame (if present)
// and uses space or f (forward) to save the location to a file
#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "zmsin.hpp"

using namespace cv;
using namespace std;
using namespace boost::filesystem;

void mouseCallback(int event, int x, int y, int flags, void *userData)
{
	Rect *r = static_cast<Rect *>(userData);
	Point p = r->tl();

	// TODO create callback to handle mouse interaction
	// event == CV_EVENT_MOUSEDOWN - 
	//    reset rect tl() to coords x,y, set width & height to 0
	// (event == CV_EVENT_MOUSEMOVE) && (event & CV_EVENT_FLAG_LBUTTON)
	//    Create point = current rect coords
	//    r->x = min(point.x, x); and same for y
	//    r->width = abs(point.x-x); and same for y
	//
	//    Maybe add right click to clear rect from screen - set x and y to -1?
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		cerr << "mark_goals input_file.zms" << endl;
		return -1;
	}
	MediaIn *cap = new ZMSIn(argv[1]);
	path fileName(argv[1]);
	cout << "Opened " << fileName.filename().string() << endl;

	if (!cap->isOpened())
	{
		cerr << "Error opening input file" << endl;
		return -1;
	}

	// TODO Open ofstream "goal_truth.txt" for append in text mode


	namedWindow("Image");
	Rect r(-1, -1, 0, 0);
	setMouseCallback("Image", mouseCallback, &r);

	// TODO create var to hold selected rectangle 
	// TODO hook up mouse callback

	Mat image;
	bool quit = false;
	 
	while (!quit && cap->getFrame(image, true))
	{
		imshow ("Image", image);

		if (r.tl() != Point(-1,-1))
		{
			// Draw rectangle on screen
		}

		int ch = waitKey(5);
		switch (ch)
		{
			case ' ':
			case 'f':
				// TODO Write a line to goal_truth.txt if a rectangle is selected:
				// <input name> <frame number> <bounding rect x, y, width, height>

				// Read next frame, exit on end of file
				if (!cap->getFrame(image, false))
				{
					quit = true;
					break;
				}
				cout << "Frame " << cap->frameNumber() << endl;

				//TODO clear out selection rectangle by setting x&y to -1?

				break;
			case 27:
			case 'q':
				quit = true;
				break;
		}
	}
	delete cap;
	return 0;
}
