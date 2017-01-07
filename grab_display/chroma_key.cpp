#include <vector>
#include <opencv2/opencv.hpp>
#include "chroma_key.hpp"
#include "opencv2_3_shim.hpp"
#if CV_MAJOR_VERSION == 2
#define cuda gpu
#else
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#endif

//#define DEBUG
using namespace std;
using namespace cv;
const static int min_area = 7000;
// Given an input frame, look for an object surrounded by
// the chroma key color. If found, return a vector of 
// points describing the contour of that object.
// If not found, return empty vector.
vector<Point> FindObjPoints(const Mat &frame, const Scalar &rangeMin, const Scalar &rangeMax)
{
    Mat objMask;

    inRange(frame, rangeMin, rangeMax, objMask);
#ifdef DEBUG
    imshow("objMask in FindRect", objMask);
#endif
    vector<vector<Point> > contours;
    vector<Vec4i>          hierarchy;
    findContours(objMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int savedContour = -1;
	
	for (size_t i = 0; i < hierarchy.size(); i++)
	{
		if ((hierarchy[i][3] >= 0) && (boundingRect(contours[i]).area() > min_area))
		{
			if (savedContour != -1)
			{
				// Multiple valid contours means the
				// code is likely finding noise.
				// Return and empty contour to indicate
				// this frame isn't a good one to use
				return vector<Point>();
			}
			savedContour = i;
		}
	}
	if (savedContour != -1)
		return contours[savedContour];
	return vector<Point>();
}

bool FindRect(const Mat& frame, const Scalar &rangeMin, const Scalar &rangeMax, Rect& output)
{
    /* prec: frame image &frame, integers 0-255 for each min and max HSV value
     *  postc: a rectangle bounding the image we want
     *  takes the frame image, filters to only find values in the range we want, finds
     *  the counters of the object and bounds it with a rectangle
     */
    vector<Point> points = FindObjPoints(frame, rangeMin, rangeMax);
    if (points.empty())
    {
        return false;
    }
    output = boundingRect(points);
    return true;
}

// Return a mask image. It will be the same
// size as the input. It looks for a solid area of color
// in the h,s,v range specified with an object in the
// middle. Pixels corresponding to the object location
// will be set to 255 in the mask, all others will be 0
// Also return the bounding rect for the object since
// we have everything we need to calculate it
bool getMask(const Mat &frame, const Scalar &rangeMin, const Scalar &rangeMax, Mat &objMask, Rect &boundRect)
{
    vector<Point> points = FindObjPoints(frame, rangeMin, rangeMax);
    if (points.empty())
    {
        return false;
    }
    boundRect = boundingRect(points);
	objMask = Mat::zeros(frame.size(), CV_8UC1);
#ifdef DEBUG
	imshow("getMask initial objMask", objMask);
#endif
    vector<vector<Point> > contours;
	contours.push_back(points);
	drawContours(objMask, contours, 0, Scalar(255), CV_FILLED);
#ifdef DEBUG
	imshow("getMask drawContours objMask", objMask);
#endif
	int dilation_type = MORPH_ELLIPSE;
	int dilation_size = 1;
	Mat element       = getStructuringElement(dilation_type,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size));
	dilate(objMask, objMask, element);
#ifdef DEBUG
	imshow("dilate objMask", objMask);
#endif
	int erosion_size = 5;
	element = getStructuringElement(dilation_type,
			Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			Point(erosion_size, erosion_size));
	erode(objMask, objMask, element);
#ifdef DEBUG
	imshow("erode objMask", objMask);
#endif
	return true;
}

GpuMat doChromaKey(const GpuMat &fgImage, const GpuMat &bgImage, const GpuMat &mask)
{
	vector<GpuMat> channels;

	// Add mask as 4th alpha (transparency) channel to fgImage
	GpuMat fgRGBA;
	cuda::split(fgImage, channels);
	channels.push_back(mask);
	cuda::merge(channels, fgRGBA);

	// Add inverse of mask as 4th alpha (transparency) channel to bgImage
	GpuMat bgRGBA;
	cuda::resize(bgImage, bgRGBA, fgImage.size());
	cuda::split(bgRGBA, channels);
	GpuMat maskInv;
	cuda::bitwise_not(mask, maskInv);
	channels.push_back(maskInv);
	cuda::merge(channels, bgRGBA);

	GpuMat dst;

	// Do alpha composition with fg on top of background
	alphaComp(fgRGBA, bgRGBA, dst, cuda::ALPHA_OVER);

	return dst;
}

// Modified from https://gist.github.com/enpe/8634ce7f200fb554f0e5
Mat doChromaKey(const Mat &fgImage, const Mat &bgImage, const Mat &mask)
{
	const Size imageSize = fgImage.size();
	Mat newImage;
	resize(bgImage, newImage, imageSize);

	for ( int y = 0; y < imageSize.height; ++y )
	{
		const uchar *ptrMask = mask.ptr<uchar>(y);
		const Vec3b *ptrFg   = fgImage.ptr<Vec3b>(y);
		      Vec3b *ptrNew  = newImage.ptr<Vec3b>(y);
		for ( int x = 0; x < imageSize.width; ++x )
		{
			uint8_t maskValue = ptrMask[x];

			if ( maskValue == 255 )
			{
				ptrNew[x] = ptrFg[x];
			}
			else if ( maskValue != 0 )
			{
				const double alpha = 1. / static_cast< double >( maskValue );
				ptrNew[x] = alpha * ptrFg[x] + ( 1. - alpha ) * ptrNew[x];
			}
			// mask == 0 -> use the bg image. This is already the
			// case since bgImg was resized into newImgage
			// outside the loop
		}
	}

	return newImage;
}
