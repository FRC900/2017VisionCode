#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <opencv2/opencv.hpp>
#include "utilities_common.h"

using namespace std;
using namespace cv;

int main(int argc, const char* argv[])
{
	// Output folder name
    const string oFolder = "/home/kjaget/CNN_DEMO/negative/generic";
	if (mkdir(oFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
	{
		if (errno != EEXIST)
		{
			cerr << "Could not create " << oFolder.c_str() << ":";
			perror("");
			return -1;
		}
	}
    
	// Input folders to grab negatives from
    vector<string> filePaths;
    GetFilePaths("/media/kjaget/disk/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
    GetFilePaths("/media/kjaget/disk/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
    GetFilePaths("/media/kjaget/disk/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
    cout << filePaths.size() << " images!" << endl;

    RNG rng( time (NULL) );
    const int nNegs = 50000;

    Mat img;    // the full image
    Mat rsz;    // the randomly picked subimage
    
	// Use high (still lossless) compression to save disk
	// space - there are lots of images so any small savings will help
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    for (int nDone = 0; nDone < nNegs; ) 
	{
		// Grab a random image from the list
		size_t idx = rng.uniform(0, filePaths.size());
		img = imread(filePaths[idx]);

		// Pick a random row and column from the image
		int r = rng.uniform(0, img.rows);
		int c = rng.uniform(0, img.cols);

		// Pick a random size as well. Make sure it
		// doesn't extend past the edge of the input
		int s = rng.uniform(0, 2*MIN(MIN(r, img.rows-r), MIN(c, img.cols-c)) + 1 );

		if (s < 28)
			continue;

		Rect rect = Rect(c-s/2, r-s/2, s, s);
		cv::resize(img(rect), rsz, cv::Size(48, 48));
		stringstream ss;
		ss << oFolder << "/" << filePaths[idx].substr(filePaths[idx].find_last_of("\\/") + 1) << "_" << setfill('0') << "_" << setw(4) << rect.x << "_" << setw(4) << rect.y << "_" << setw(4) << rect.width << "_" << setw(4) << rect.height << ".png";
		//cerr << ss.str() << endl;
		imwrite(ss.str(), rsz, compression_params);
		nDone++;
		if (nDone % 1000 == 0)
			cout << nDone << " neg generated!" << endl;
	}
    
    return 0;
}
