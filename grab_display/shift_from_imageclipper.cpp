// Take a dir full of output images grabbed by
// hand using imageclipper
// Decode the bounding rect from the fileame
// Go back to the original video and generate
// shifted versions of the rects for training
// calibration nets
#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imageShift.hpp"
#include "imageclipper_read.hpp"

using namespace std;
using namespace cv;

string srcPath = "/home/kjaget/ball_videos/white_floor/";
string outPath = "shifts";
int main(void)
{
	const double targetAR = 1.0;
	DIR *dirp = opendir(".");
	struct dirent *dp;
	vector<string> image_names;
	if (!dirp)
		return -1;

	while ((dp = readdir(dirp)) != NULL) 
	{
		if (strstr(dp->d_name, ".png") )
			image_names.push_back(dp->d_name);
	}
	closedir(dirp);
	cout << "Read " << image_names.size() << " image names" << endl;

	createShiftDirs(outPath);
	RNG rng(time(NULL));
	Mat mat;
	Rect rect;
	string inFileName;
	int frame;
	for (vector<string>::iterator it = image_names.begin(); it != image_names.end(); ++it)
	{
		if (getFrameAndRect(*it, srcPath, targetAR, inFileName, frame, mat, rect))
		{

			stringstream write_name;
			write_name << inFileName;
			write_name << "_" << setw(5) << setfill('0') << frame;
			write_name << "_" << setw(4) << rect.x;
			write_name << "_" << setw(4) << rect.y;
			write_name << "_" << setw(4) << rect.width;
			write_name << "_" << setw(4) << rect.height;
			write_name << ".png";

			doShifts(mat, rect, rng, Point3f(0,0,2.0*M_PI), 4, outPath, write_name.str());
		}
	}
	return 0;
}

