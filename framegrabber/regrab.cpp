// Code to re-grab images grabbed by imageclipper
// Make sure the re-grabbed image is exactly the aspect
// ratio wanted.
// Apply different random noise to several copies of the output
// Optionally created resized versions of the output as well
#include <sys/types.h>
#include <dirent.h>
#include "opencv2/opencv.hpp"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "cv.h"
#include "cvaux.h"
#include "cxcore.h"
#include "highgui.h"
#include "opencvx/cvrect32f.h"
#include "opencvx/cvdrawrectangle.h"
#include "opencvx/cvcropimageroi.h"
#include "opencvx/cvpointnorm.h"
using namespace std;
using namespace cv;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		if (!item.empty())
			elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}
const double targetAR = 1.0;

int main(void)
{
	DIR *dirp = opendir(".");
	struct dirent *dp;
	vector<string> image_names;
	if (!dirp)
		return -1;

	while ((dp = readdir(dirp)) != NULL) {
		if (strstr(dp->d_name, ".png") && !strstr(dp->d_name, "_r.png"))
			image_names.push_back(dp->d_name);
	}
	closedir(dirp);
	cout << "Read " << image_names.size() << " image names" << endl;

	RNG rng(time(NULL));
	const char *w_name = "1";
	const char *miniw_name = "2";
	CvPoint shear = Point(0,0);
	//cvNamedWindow( w_name, CV_WINDOW_AUTOSIZE );
	//cvNamedWindow( miniw_name, CV_WINDOW_AUTOSIZE );
	for (vector<string>::iterator it = image_names.begin(); it != image_names.end(); ++it)
	{
		int frame;
		int rotation;
		Rect rect;
		*it = it->substr(0, it->rfind('.'));
		vector<string> tokens = split(*it, '_');
		frame       = atoi(tokens[tokens.size()-6].c_str());
		rotation    = atoi(tokens[tokens.size()-5].c_str());
		rect.x      = atoi(tokens[tokens.size()-4].c_str());
		rect.y      = atoi(tokens[tokens.size()-3].c_str());
		rect.width  = atoi(tokens[tokens.size()-2].c_str());
		rect.height = atoi(tokens[tokens.size()-1].c_str());

		string inFileName = "../";
		for (size_t i = 0; i < tokens.size()-6; i++)
		{
			inFileName += tokens[i];
			if (i < tokens.size() - 7)
				inFileName += "_";
		}

		CvCapture *cap = cvCaptureFromFile( inFileName.c_str() );
		cvSetCaptureProperty( cap, CV_CAP_PROP_POS_FRAMES, frame - 1 );
		IplImage *img = cvQueryFrame( cap );
		if( img == NULL )
		{
			cerr << "Can not open " << inFileName << endl;
			cvReleaseCapture(&cap);
			continue;
		}
		//cvDrawRectangle(img,cvRect32fFromRect(rect, rotation), cvPointTo32f(shear)); 
		double ar = rect.width / (double)rect.height;
		int added_height = 0;
		int added_width  = 0;
		if (ar > targetAR)
		{
			added_height = rect.width / targetAR - rect.height;
			rect.x -= (added_height/ 2.0) * sin(rotation / 180.0 * M_PI);
			rect.y -= (added_height/ 2.0) * cos(rotation / 180.0 * M_PI);
		}
		else
		{
			added_width = rect.height * targetAR - rect.width;
			rect.x -= (added_width / 2.0) * cos(rotation / 180.0 * M_PI);
			rect.y += (added_width / 2.0) * sin(rotation / 180.0 * M_PI);
		}
		rect.width  += added_width;
		rect.height += added_height;

#if 0
		//cout << rect.width << " " << rect.height << " " << added_width << " "<< added_height <<endl;

		cvShowImageAndRectangle( w_name, img, 
				cvRect32fFromRect( rect, rotation), 
				cvPointTo32f( shear )); 
		cvShowCroppedImage( miniw_name, img, 
				cvRect32fFromRect( rect, rotation ), 
				cvPointTo32f( shear ));
#endif

		for (double size = 0.00; size <= 0.01; size += 0.07)
		{
			double dSize = rect.width * size;
			Rect thisRect = rect;
			thisRect -= Point(cvRound(dSize/2), cvRound(dSize/2));
			thisRect += Size(cvRound(dSize), cvRound(dSize));

			if ((thisRect.x < 0) || (thisRect.y < 0) || 
				(thisRect.br().x >= img->width) || (thisRect.br().y >= img->height))
				break;

			IplImage* crop = cvCreateImage( 
					cvSize( thisRect.width, thisRect.height ), 
					img->depth, img->nChannels );
			cvCropImageROI( img, crop, 
					cvRect32fFromRect( thisRect, rotation ), 
					cvPointTo32f( shear ) );

			for (int j = 0; j < 2; j++)
			{
#if CV_MAJOR_VERSION == 2
				Mat mat(crop, true);
#else
				Mat mat = cvarrToMat(crop);
#endif

				Mat noise(mat.size(), CV_64FC1);
				Mat splitMat[3];

				mat.convertTo(mat, CV_64FC3);
				split(mat, splitMat);

				for (int i = 0; i < 3; i++)
				{
					double min, max;
					randn(noise, 0.0, 5.0);
					minMaxLoc(splitMat[i], &min, &max, NULL, NULL);
					add(splitMat[i], noise, splitMat[i]);
					normalize(splitMat[i], splitMat[i], min, max, NORM_MINMAX, -1);
				}
				merge(splitMat, 3, mat);
				mat.convertTo(mat, CV_8UC3);

				stringstream write_name;
				write_name << inFileName;
				write_name << "_" << setw(5) << setfill('0') << frame;
				write_name << "_" << setw(4) << thisRect.x;
				write_name << "_" << setw(4) << thisRect.y;
				write_name << "_" << setw(4) << thisRect.width;
				write_name << "_" << setw(4) << thisRect.height;
				write_name << "_" << setw(2) << j;
				write_name << "_r.png";
				cerr << write_name.str() << endl;
				imwrite( write_name.str().c_str(), mat );
			}
			cvReleaseImage( &crop );
		}
		cvReleaseCapture( &cap );
	}
	return 0;
}

