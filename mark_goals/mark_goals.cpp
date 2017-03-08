// Tool to create goal_truth.txt file. This file holds information on the
// locations of goals in an input video file.  The main zv executable can 
// use this file to automatically test how well goal detection is working
// The input file is a ZMS video - this includes RGB plus depth data
// The user highlights the location of a goal in a frame (if present)
// and uses space or f (forward) to save the location to a file
#include <iostream>
#include <opencv2/opencv.hpp>

#include "zmsin.hpp"

using namespace cv;
using namespace std;

// TODO create vars to hold selected rectangle 
// TODO create callback to handle mouse interaction

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		cerr << "mark_goals input_file.zms" << endl;
		return -1;
	}
	MediaIn *cap = new ZMSIn(argv[1]);

	if (!cap->isOpened())
	{
		cerr << "Error opening input file" << endl;
		return -1;
	}

	// TODO Open ofstream "goal_truth.txt" for append in text mode


	// TODO hook up mouse callback

	Mat image;
	bool quit = false;
	 
	while (!quit && cap->getFrame(image, true))
	{
		imshow ("Image", image);


		int ch = waitKey(5);
		switch (ch)
		{
			case ' ':
			case 'f':
				// TODO Write a line to goal_truth.txt if a rectangle is selected:
				// <input name> <frame number> <bounding rect x, y, width, height>
				// cap->frameNumber gets the current frame number

				// Read next frame
				if (!cap->getFrame(image, false))
				{
					quit = true;
					break;
				}
				cout << "Frame " << cap->frameNumber() << endl;

				//TODO clear out selection rectangle?

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
