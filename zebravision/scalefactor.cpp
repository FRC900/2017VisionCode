#include "opencv2_3_shim.hpp"

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#define cuda gpu
#elif CV_MAJOR_VERSION == 3
#include <opencv2/cudawarping.hpp>
using namespace cv::cuda;
#endif

static void resizeStub(const Mat &src, Mat &dest, const Size &size, const double fx = 0., const double fy = 0.)
{
	cv::resize(src, dest, size, fx, fy);
}

static void resizeStub(const GpuMat &src, GpuMat &dest, const Size &size, const double fx = 0., const double fy = 0.)
{
	cuda::resize(src, dest, size, fx, fy);
}

template <class MatT>
void scalefactor(const MatT &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<MatT, double> > &scaleInfo)
{
	scaleInfo.clear();
	/*
	Loop multiplying the image size by the scalefactor upto the maxsize	
	Store each image in the images vector
	Store the scale factor in the scales vector 
	*/

	//only works for square image?
	double scale = (double)objectsize.width / minsize.width;

	while(scale > (double)objectsize.width / maxsize.width)
	{	
		//set objectsize.width to scalefactor * objectsize.width
		//set objectsize.height to scalefactor * objectsize.height
		MatT outputimage;
		resizeStub(inputimage, outputimage, Size(), scale, scale);

		// Resize will round / truncate to integer size, recalculate
		// scale using actual results from the resize
		double newscale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		
		scaleInfo.push_back(make_pair(outputimage, newscale));

		scale /= scaleFactor;		
	}	
}


// Create an array of images which are resized from scaleInfoIn by a factor
// of resizeFactor. Used to create a list of d24-sized images from a d12 list. Can't
// use the function above with a different window size since rounding errors will add +/- 1 to
// the size versus just doing 2x the actual size of the d12 calculations
template <class MatT>
void scalefactor(const MatT &inputimage, const vector<pair<MatT, double> > &scaleInfoIn, int rescaleFactor, vector<pair<MatT, double> > &scaleInfoOut)
{
	scaleInfoOut.clear();
	for (auto it = scaleInfoIn.cbegin(); it != scaleInfoIn.cend(); ++it)
	{
		MatT outputimage;

		Size newSize(it->first.cols * rescaleFactor, it->first.rows * rescaleFactor);
		resizeStub(inputimage, outputimage, newSize);
		// calculate scale from actual size, which will
		// include rounding done to get to integral number
		// of pixels in each dimension
		double scale = max((double)outputimage.rows / inputimage.rows, (double)outputimage.cols / inputimage.cols);
		scaleInfoOut.push_back(make_pair(outputimage, scale));
	}
}

template void scalefactor(const Mat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<Mat, double> > &scaleInfo);
template void scalefactor(const GpuMat &inputimage, const Size &objectsize, const Size &minsize, const Size &maxsize, double scaleFactor, vector<pair<GpuMat, double> > &scaleInfo);
template void scalefactor(const Mat &inputimage, const vector<pair<Mat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<Mat, double> > &scaleInfoOut);
template void scalefactor(const GpuMat &inputimage, const vector<pair<GpuMat, double> > &scaleInfoIn, int rescaleFactor, vector<pair<GpuMat, double> > &scaleInfoOut);
