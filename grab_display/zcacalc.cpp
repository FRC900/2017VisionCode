#include <string>
#include "zca.hpp"

#include "random_subimage.hpp"
#include "utilities_common.h"

using namespace std;
using namespace cv;

static void doZCA(const vector<Mat> &images, const Size &size, const float epsilon, const string &id, const int seed)
{
	cout << "epsilon " << epsilon << endl;
	ZCA zca(images, size, epsilon);

	stringstream name;
	name << "zcaWeights" << id << "_" << size.width << "_" << seed << "_" << images.size();
	zca.Write(name.str() + ".xml");
	zca.WriteCompressed(name.str() + ".zca");
}

// returns true if the given 3 channel image is B = G = R
bool isGrayImage(const Mat &img) 
{
    Mat dst;
    Mat bgr[3];
    split( img, bgr );
    absdiff( bgr[0], bgr[1], dst );

    if(countNonZero( dst ))
        return false;

    absdiff( bgr[0], bgr[2], dst );
    return !countNonZero( dst );
}

vector<Mat> loadSubImages(const int seed, const int nImgs)
{
	vector<Mat> images;
	vector<string> filePaths;
	GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/framegrabber", ".png", filePaths);
	GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/Framegrabber2", ".png", filePaths, true);
	GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/generic", ".png", filePaths, true);
	GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/20160210", ".png", filePaths, true);
	GetFilePaths("/media/kjaget/AC8612CF86129A42/cygwin64/home/ubuntu/2015VisionCode/cascade_training/negative_images/white_bg", ".png", filePaths, true);
	cout << filePaths.size() << " images!" << endl;
	RandomSubImage rsi(RNG(seed), filePaths);
	Mat img; // full image data
	Mat patch; // randomly selected image patch from full image
	for (int nDone = 0; nDone < nImgs; ) 
	{
		img = rsi.get(1.0, 0.05);
		resize(img, patch, Size(48,48));
		// There are grayscale images in the 
		// negatives, but we'll never see one
		// in real life. Exclude those for now
		if (isGrayImage(img))
			continue;
		images.push_back(patch.clone());
		nDone++;
		if (!(nDone % 1000))
			cout << nDone << " image patches extracted" << endl;
	}
	return images;
}

int main(void)
{
	const int seed = 12345;
	vector<Mat> images = loadSubImages(seed, 20000);

	doZCA(images, Size(12,12), 0.001, "_newE001", seed);
	doZCA(images, Size(24,24), 0.001, "_newE001", seed);
	//doZCA(images, Size(48,48), 0.001, "_newE001", seed);
}
