// Take a dir full of output images grabbed by
// hand using imageclipper
// Decode the bounding rect from the fileame
// Go back to the original video and generate
// a given number of randomly rotated versions
// of the original image
#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image_warp.hpp"
#include "imageclipper_read.hpp"

using namespace std;
using namespace cv;

int main(void)
{
	const string srcPath = "/home/kjaget/ball_videos/white_floor/";
	const string outPath = "rotates_resize";
	const int rotateCount = 15;
	const double targetAR = 1.0;
	const int minResize = 0;
	const int maxResize = 25;
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

	RNG rng(time(NULL));
	Mat mat;
	Rect rect;
	int frame;
	string inFileName;
	Mat rotImg;
	Mat rotMask;
	for (vector<string>::iterator it = image_names.begin(); it != image_names.end(); ++it)
	{
		//cout << *it << endl;
		if (getFrameAndRect(*it, srcPath, targetAR, inFileName, frame, mat, rect))
		{
			stringstream write_name;
			write_name << inFileName;
			write_name << "_" << setw(5) << setfill('0') << frame;
			write_name << "_" << setw(4) << rect.x;
			write_name << "_" << setw(4) << rect.y;
			write_name << "_" << setw(4) << rect.width;
			write_name << "_" << setw(4) << rect.height;

			// create another rect expanded to the limits of the input 
			// image size with the object still in the center.
			// This will allow us to get the pixels from a corner as
			// the image is rotated
#if 0
			cout << mat.size() << endl;
			cout << rect << endl;
#endif
			int added_x = min(rect.tl().x, mat.cols - 1 - rect.br().x);
			int added_y = min(rect.tl().y, mat.rows - 1 - rect.br().y);
			int added_size = min(added_x, int(added_y * targetAR));
			//cout << added_x << " " << added_y << " " << added_size << endl;
			Rect largeRect(rect.tl() - Point(added_size, added_size/targetAR), 
					       rect.size() + Size(2*added_size, 2*int(added_size/targetAR)));
			Rect newOrigRect(added_size, added_size/targetAR, rect.width, rect.height);
#if 0
			cout << largeRect << endl;
			cout << newOrigRect << endl;
			imshow("Mat", mat);
			imshow("rect", mat(rect));
			imshow("largeRect", mat(largeRect));
			imshow("newOrigRect",mat(largeRect)(newOrigRect));
#endif
			int failCount = 0;
			for (int i = 0; (i < rotateCount) && (failCount < 100); )
			{
				Rect finalRect;
				if (RescaleRect(newOrigRect, finalRect, Size(largeRect.width, largeRect.height), rng.uniform(minResize, maxResize)))
				{

					rotateImageAndMask(mat(largeRect), Mat(), Scalar(mat(largeRect).at<Vec3b>(0,0)), Point3f(0,0,M_PI*2.0), rng, rotImg, rotMask);
					stringstream s;
					s << outPath << "/" << write_name.str() << "_" << setw(2) << setfill('0') << i << ".png";
					//			cout << s.str() << endl;
					imwrite(s.str(), rotImg(finalRect));
					i++;
					failCount = 0;
				}
				else
				{
					failCount++;
				}
			}
			//waitKey(0);
		}
	}
	return 0;
}

