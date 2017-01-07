#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#include "chroma_key.hpp"
#include "image_warp.hpp"
#include "random_subimage.hpp"
#include "opencv2_3_shim.hpp"
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#define cuda gpu
#elif CV_MAJOR_VERSION == 3
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
using namespace cv::cuda;
#endif

using namespace std;
using namespace cv;

// x, y and size shift values
static const float dx = .17;
static const float dy = .17;
static const float ds[5] = {.83, .91, 1.0, 1.10, 1.21};

static Rect shiftRect(const Rect rectIn, float ds, float dx, float dy)
{
	return Rect(cvRound(rectIn.tl().x - (dx*rectIn.width /ds)), 
			cvRound(rectIn.tl().y - (dy*rectIn.height/ds)), 
			cvRound(rectIn.width /ds),
			cvRound(rectIn.height/ds));
}

// Create the various output dirs - the base shift
// directory and directories numbered 0 - 44.
bool createShiftDirs(const string &outputDir)
{
	if (mkdir(outputDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
	{
		if (errno != EEXIST)
		{
			cerr << "Could not create " << outputDir.c_str() << ":";
			perror("");
			return false;
		}
	}
	// Create output directories
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				string dir_name = to_string(is*9 + ix*3 + iy);
				if (mkdir((outputDir+"/"+dir_name).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH))
				{
					if (errno != EEXIST)
					{
						cerr << "Could not create " << (outputDir+"/"+dir_name).c_str() << ":";
						perror("");
						return false;
					}
				}
			}
		}
	}
	return true;
}
// For some reason the code can't figure out Mat vs. GpuMat
// versions of these. Hard-code Mat vs. GpuMat ones to call
// the cv:: vs. cuda:: versions manually
bool imwriteStub(const string &fileName, const Mat &mat)
{
	return imwrite(fileName,mat);
}

bool imwriteStub(const string &fileName, const GpuMat &gpuMat)
{
	Mat mat;
	gpuMat.download(mat);
	return imwrite(fileName,mat);
}
static void resizeStub(const Mat &src, Mat &dest, const Size &size, const double fx = 0., const double fy = 0.)
{
	cv::resize(src, dest, size, fx, fy);
}

static void resizeStub(const GpuMat &src, GpuMat &dest, const Size &size, const double fx = 0., const double fy = 0.)
{
	cuda::resize(src, dest, size, fx, fy);
}

static void copyMakeBorderStub(const Mat &src, Mat &dst, int top, int bottom, int left, int right, int borderType, const Scalar &value = Scalar() )
{
	cv::copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);
}

static void copyMakeBorderStub(const GpuMat &src, GpuMat &dst, int top, int bottom, int left, int right, int borderType, const Scalar &value = Scalar() )
{
	// cuda copyMakeBorder only handles 1 or 4 channel uchar 
	// images. Add a bogus alpha channel here to convert
	// 3-channel inputs to 4 channels
	GpuMat localSrc;
	if (src.type() == CV_8UC3)
		cuda::cvtColor(src, localSrc, COLOR_BGR2BGRA);
	else
		localSrc = src;

	cuda::copyMakeBorder(localSrc, dst, top, bottom, left, right, borderType, value);

	if (src.type() == CV_8UC3)
	{
		GpuMat m;
		cuda::cvtColor(dst, m, COLOR_BGRA2BGR);
		m.copyTo(dst);
	}
}

// Given a src image and an object ROI within that image,
// generate shifted versions of the object
template <class MatT>
static void doShiftsInternal(const MatT &src,   // source image
			const Rect &objROI, // ROI of object within that image
			const Scalar &fillColor, // color to fill in on new pixels
			RNG &rng,           // random num generator
			const Point3f &maxRot, // max rotation in XYZ in radians
			int copiesPerShift,    // number of randomly rotated copies to make
			const string &outputDir, // base output dir
			const string &fileName)  // base out filename
{
	MatT rotImg;  // randomly rotated input
	MatT rotMask; // and mask
	MatT final;   // final output

	if (src.empty())
	{
		return;
	}

	// strip off directory and .png suffix
	string fn(fileName);
	auto pos = fn.rfind('/');
	if (pos != std::string::npos)
	{
		fn.erase(0, pos + 1);
	}
	pos = fn.rfind('.');
	if (pos != std::string::npos)
	{
		fn.erase(pos);
	}
	cout << fn << endl;

	// create another rect expanded to the limits of the input
	// image size with the object still in the center.
	// This will allow us to save the pixels from a corner as
	// the image is rotated
	const double targetAR = (double) objROI.width / objROI.height;
	const int added_x = min(objROI.tl().x, src.cols - 1 - objROI.br().x);
	const int added_y = min(objROI.tl().y, src.rows - 1 - objROI.br().y);
	const int added_size = min(added_x, int(added_y * targetAR));
	const Rect largeRect(objROI.tl() - Point(added_size, added_size/targetAR),
				   objROI.size() + Size(2*added_size, 2*int(added_size/targetAR)));

	const Rect largeRectBounds(0,0,largeRect.width, largeRect.height);
	// This is a rect which will be the input objROI but
	// in coorindates relative to the largeRect created above
	const Rect newObjROI(added_size, added_size/targetAR, objROI.width, objROI.height);
	// Generate copiesPerShift images per shift/scale permutation
	// So each call will end up with 5 * 3 * 3 * copiesPerShift
	// images writen
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				for (int c = 0; c < copiesPerShift; c++)
				{
					// Shift/rescale the region of interest based on
					// which permuation of the shifts/rescales we're at
					const Rect thisROI = shiftRect(newObjROI, ds[is], (ix-1)*dx, (iy-1)*dy);
					if ((largeRectBounds & thisROI) != thisROI)
					{
						cerr << "Rectangle out of bounds for " << is 
							<< " " << ix << " " << iy << " " << 
							largeRectBounds.size() << " vs " << thisROI << endl;
						break;
					}

					// Rotate the image a random amount.  Mask isn't used
					// since there's no chroma-keying going on.
					rotateImageAndMask(src(largeRect), MatT(), fillColor, maxRot, rng, rotImg, rotMask);

#if 0
					rotImg(thisROI).copyTo(final);
					imshow("src", src);
					imshow("src(objROI)", src(objROI));
					imshow("src twice ROI", src(twiceObjROI));
					imshow("src(twice ROI)(newObjROI)", src(twiceObjROI)(newObjROI));
					imshow("rotImg", rotImg);
					imshow("rotImg(newObjROI)", rotImg(newObjROI));
					resize (final, final, Size(240,240));
					imshow("Final", final);
					waitKey(0);
#else
					// 48x48 is the largest size we'll need from here on out,
					// so resize to that to save disk space
					resizeStub(rotImg(thisROI), final, Size(48,48));
#endif

					// Dir name is a number from 0 - 44.
					// 1 per permutation of x,y shift plus resize
					string dir_name = to_string(is*9 + ix*3 + iy);
					string write_file = outputDir + "/" + dir_name + "/" + fn + "_" + to_string(c) + ".png";
					if (imwriteStub(write_file, final) == false)
					{
						cout << "Error! Could not write file "<<  write_file << endl;
					}
				}
			}
		}
	}
}

template <class MatT>
static void doShiftsInternal(const MatT &src,  // tightly cropped image of object
			const MatT &mask,   // binary mask of object/not-object pixels
			const Scalar &fillColor, // used to fill in new pixels
			RNG &rng,          // random number generator
			RandomSubImage &rsi, // random background image class
			const Point3f &maxRot, // max rotation in XYZ in radians
			int copiesPerShift,    // number of randomly rotated images 
			const string &outputDir, // base output dir
			const string &fileName)  // base filename
{
	MatT original;
	MatT objMask;
	MatT bgImg;    // random background image to superimpose each input onto 
	MatT rotImg;  // randomly rotated input
	MatT rotMask; // and mask
	MatT chromaImg; // combined input plus bg
	MatT final;   // final output

	if (src.empty() || mask.empty())
	{
		return;
	}

	// strip off directory and .png suffix
	string fn(fileName);
	auto pos = fn.rfind('/');
	if (pos != std::string::npos)
	{
		fn.erase(0, pos + 1);
	}
	pos = fn.rfind('.');
	if (pos != std::string::npos)
	{
		fn.erase(pos);
	}
	cout << fn << endl;

	// Enlarge the original image.  Since we're shifting the region
	// of interest need to do this to make sure we don't end up 
	// outside the mat boundries
	const int expand = max(src.rows, src.cols) / 2;
	const Rect origROI(expand, expand, src.cols, src.rows);
	copyMakeBorderStub(src, original, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(fillColor));
	copyMakeBorderStub(mask, objMask, expand, expand, expand, expand, BORDER_CONSTANT, Scalar(0));
	
	// Generate copiesPerShift images per shift/scale permutation
	// So each call will end up with 5 * 3 * 3 * copiesPerShift
	// images writen
	for (int is = 0; is < 5; is++)
	{
		for (int ix = 0; ix <= 2; ix++)
		{
			for (int iy = 0; iy <= 2; iy++)
			{
				for (int c = 0; c < copiesPerShift; c++)
				{
					// Rotate the image a random amount. Also rotat the mask
					// so they stay in sync with each other
					rotateImageAndMask(original, objMask, fillColor, maxRot, rng, rotImg, rotMask);

					// Get a random background image, superimpose
					// the object on top of that image
					bgImg = MatT(rsi.get((double)original.cols / original.rows, 0.05));

					chromaImg = doChromaKey(rotImg, bgImg, rotMask);

					// Shift/rescale the region of interest based on
					// which permuation of the shifts/rescales we're at
					const Rect ROI = shiftRect(origROI, ds[is], (ix-1)*dx, (iy-1)*dy);
#if 0
					chromaImg(ROI).copyTo(final);
					imshow("original", original);
					imshow("bgImg", bgImg);
					imshow("rotImg", rotImg);
					imshow("rotMask", rotMask);
					imshow("chromaImg", chromaImg);
					resize (final, final, Size(240,240));
					imshow("Final", final);
					waitKey(0);
#else
					// 48x48 is the largest size we'll need from here on out,
					// so resize to that to save disk space
					resizeStub(chromaImg(ROI), final, Size(48,48));
#endif

					// Dir name is a number from 0 - 44.
					// 1 per permutation of x,y shift plus resize
					string dir_name = to_string(is*9 + ix*3 + iy);
					string write_file = outputDir + "/" + dir_name + "/" + fn + "_" + to_string(c) + ".png";
					if (imwriteStub(write_file, final) == false)
					{
						cout << "Error! Could not write file "<<  write_file << endl;
					}
				}
			}
		}
	}
}

void doShifts(const Mat &src,   // source image
			const Rect &objROI, // ROI of object within that image
			RNG &rng,           // random num generator
			const Point3f &maxRot, // max rotation in XYZ in radians
			int copiesPerShift,    // number of randomly rotated copies to make
			const string &outputDir, // base output dir
			const string &fileName)  // base out filename
{
	const Scalar fillColor = Scalar(src(objROI).at<Vec3b>(0,0));
	if (getCudaEnabledDeviceCount() > 0)
	{
		GpuMat srcGPU(src);
		doShiftsInternal(srcGPU, objROI, fillColor, rng, maxRot, copiesPerShift, outputDir, fileName);
	}
	else
		doShiftsInternal(src, objROI, fillColor, rng, maxRot, copiesPerShift, outputDir, fileName);
}

void doShifts(const Mat &src,  // tightly cropped image of object
			const Mat &mask,   // binary mask of object/not-object pixels
			RNG &rng,          // random number generator
			RandomSubImage &rsi, // random background image class
			const Point3f &maxRot, // max rotation in XYZ in radians
			int copiesPerShift,    // number of randomly rotated images 
			const string &outputDir, // base output dir
			const string &fileName)  // base filename
{
	// Use color at 0,0 to fill in expanded rect assuming that
	// location is the chroma-key color for that given image
	// Probably want to pass this in instead for cases
	// where we're working from a list of files captured from live
	// video rather than video shot against a fixed background - can't
	// guarantee the border color there is safe to use
	const Scalar fillColor = Scalar(src.at<Vec3b>(0,0));

	if (getCudaEnabledDeviceCount() > 0)
	{
		GpuMat srcGPU(src);
		GpuMat maskGPU(mask);
doShiftsInternal(srcGPU, maskGPU, fillColor, rng, rsi, maxRot, copiesPerShift, outputDir, fileName);
	}
	else
		doShiftsInternal(src, mask, fillColor, rng, rsi, maxRot, copiesPerShift, outputDir, fileName);
}


template static void doShiftsInternal(const Mat &src, const Rect &objROI, const Scalar &fillColor, RNG &rng, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName);
template static void doShiftsInternal(const GpuMat &src, const Rect &objROI, const Scalar &fillColor, RNG &rng, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName);
template static void doShiftsInternal(const Mat &src,  const Mat &mask, const Scalar &fillColor, RNG &rng, RandomSubImage &rsi, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName);
template static void doShiftsInternal(const GpuMat &src,  const GpuMat &mask, const Scalar &fillColor, RNG &rng, RandomSubImage &rsi, const Point3f &maxRot, int copiesPerShift, const string &outputDir, const string &fileName);
