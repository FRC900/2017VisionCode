#include <opencv2/opencv.hpp>
#include "opencv2_3_shim.hpp"
#if CV_MAJOR_VERSION == 2
#define cuda gpu
#else
#include <opencv2/cudawarping.hpp>
#endif

using namespace std;
using namespace cv;

// Resizes a rectangle to a new size, keeping
// it centered on the same point
Rect ResizeRect(const Rect& rect, const Size& size)
{
    Point tl = rect.tl();
    tl.x = cvRound(tl.x - ((double)size.width - rect.width) / 2);
    tl.y = cvRound(tl.y - ((double)size.height - rect.height) / 2);

    return Rect(tl, size);
}


Rect AdjustRect(const Rect& rect, const double ratio)
{
    // adjusts the size of the rectangle to a fixed aspect ratio
    int width  = rect.width;
    int height = rect.height;

    if (width / ratio > height)
    {
        height = width / ratio;
    }
    else if (width / ratio < height)
    {
        width = height * ratio;
    }

    return ResizeRect(rect, Size(width, height));
}


bool RescaleRect(const Rect& inRect, Rect& outRect, const Size& imageSize, const double scaleUp)
{
    // takes the rect inRect and resizes it larger by 1+scale_up percent
    // outputs resized rect in outRect
    int width  = inRect.width * (1.0 + scaleUp / 100.0);
    int height = inRect.height * (1.0 + scaleUp / 100.0);

    outRect = ResizeRect(inRect, Size(width, height));

    if ((outRect.x < 0) || (outRect.y < 0) ||
        (outRect.br().x > imageSize.width) || (outRect.br().y > imageSize.height))
    {
        cout << "Rectangle out of bounds!" << endl;
        return false;
    }
    return true;
}

static void warpPerspectiveStub(const Mat &src, Mat &dst, const Mat &M, Size dsize, int flags, int borderMode, const Scalar &borderValue)
{
	cv::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue);
}

static void warpPerspectiveStub(const GpuMat &src, GpuMat &dst, const Mat &M, Size dsize, int flags, int borderMode, const Scalar &borderValue)
{
	cuda::warpPerspective(src, dst, M, dsize, flags, borderMode, borderValue);
}

// Warps source into destination by a perspective transform
template <class MatT>
static void warpPerspective(const MatT &src, MatT &dst, 
							const Point2f outputQuad[4], 
							const Scalar &bgColor)
{
	Point2f inputQuad[4];  // Input Quadilateral or Image plane coordinates

	inputQuad[0] = Point2f(0, 0);
	inputQuad[1] = Point2f(src.cols - 1, 0);
	inputQuad[2] = Point2f(src.cols - 1, src.rows - 1);
	inputQuad[3] = Point2f(0, src.rows - 1);

	Mat lambda = getPerspectiveTransform( inputQuad, outputQuad );

	warpPerspectiveStub(src, dst, lambda, dst.size(), 
					INTER_LINEAR, BORDER_CONSTANT, bgColor);
}


static void randomQuad(const Size &size, const Point3d &maxAngle, 
					   RNG &rng, Point2f quad[4])
{
    const double distfactor = 3.0;
    const double distfactor2 = 1.0;

    double rotVectData[3];
    double vectData[3];
    double rotMatData[9];

    double d;

    Mat rotVect(3, 1, CV_64FC1, &rotVectData[0]);
    Mat rotMat(3, 3, CV_64FC1, &rotMatData[0]);
    Mat vect(3, 1, CV_64FC1, &vectData[0]);

    rotVectData[0] = maxAngle.x * rng.uniform(-1.0, 1.0);
    rotVectData[1] = (maxAngle.y - fabs(rotVectData[0])) * rng.uniform(-1.0, 1.0);
    rotVectData[2] = maxAngle.z * rng.uniform(-1.0, 1.0);
    d = (distfactor + distfactor2 * rng.uniform(-1.0, 1.0)) * size.width;

#if 0 // Debugging - set angles to max
    rotVectData[0] = maxAngle.x;
    rotVectData[1] = maxAngle.y;
    rotVectData[2] = maxAngle.z;

    d = distfactor * size.width;
#endif

    Rodrigues(rotVect, rotMat);

    const double halfw = 0.5 * size.width;
    const double halfh = 0.5 * size.height;

    quad[0].x = -halfw;
    quad[0].y = -halfh;
    quad[1].x =  halfw;
    quad[1].y = -halfh;
    quad[2].x =  halfw;
    quad[2].y =  halfh;
    quad[3].x = -halfw;
    quad[3].y =  halfh;

    for(int i = 0; i < 4; i++)
    {
        rotVectData[0] = quad[i].x;
        rotVectData[1] = quad[i].y;
        rotVectData[2] = 0.0;
        //cvMatMulAdd(&rotMat, &rotVect, 0, &vect);
		gemm(rotMat, rotVect, 1., Mat(), 1., vect, 0);
		//vect = rotMat.t() * rotVect;
        quad[i].x = vectData[0] * d / (d + vectData[2]) + halfw;
        quad[i].y = vectData[1] * d / (d + vectData[2]) + halfh;

        /*
        quad[i][0] += halfw;
        quad[i][1] += halfh;
        */
    }
}

// Given an inputimage and chroma-key mask, rotate both
// by up to the angles (in radians) given in maxAngle.
// Fill in space created due to the rotation with bgColor
template <class MatT>
void rotateImageAndMask(const MatT &srcImg, const MatT &srcMask,
						const Scalar &bgColor, const Point3f &maxAngle,
						RNG &rng, MatT &outImg, MatT &outMask)
{
	if (maxAngle == Point3f(0,0,0))
	{
		srcImg.copyTo(outImg);
		srcMask.copyTo(outMask);
		return;
	}

	// Generate a random rotation
    Point2f quad[4];
    randomQuad(srcImg.size(), maxAngle, rng, quad);
	const Point2f srcTL(srcImg.cols / 2, srcImg.rows / 2);
	for (int i = 0; i < 4; i++)
		quad[i] += srcTL;

	// Create larger tmp Mats to hold rotated outputs
	const Size dbl(srcImg.cols * 2, srcImg.rows * 2);
	MatT imgTmp(dbl, srcImg.type()); 
	MatT maskTmp(dbl, srcMask.type());  // should always be CV_8UC1

	warpPerspective(srcImg, imgTmp, quad, bgColor);
	if (!srcMask.empty())
		warpPerspective(srcMask, maskTmp, quad, Scalar(0));

	// Maybe, maybe not?
	//cv::GaussianBlur( *data->maskimg, *data->maskimg, cv::Size(3, 3), 1.0 );

	// Grab middle of rotated images, copy to outputs
	Rect cr(srcTL, srcImg.size());

	imgTmp(cr).copyTo(outImg);
	maskTmp(cr).copyTo(outMask);
}


template void rotateImageAndMask(const Mat &srcImg, const Mat &srcMask,
						const Scalar &bgColor, const Point3f &maxAngle,
						RNG &rng, Mat &outImg, Mat &outMask);
template void rotateImageAndMask(const GpuMat &srcImg, const GpuMat &srcMask,
						const Scalar &bgColor, const Point3f &maxAngle,
						RNG &rng, GpuMat &outImg, GpuMat &outMask);
