#include <iostream>
#include "zedsvoin.hpp"
using namespace std;

#ifdef ZED_SUPPORT
using namespace cv;
using namespace sl::zed;

ZedSVOIn::ZedSVOIn(const char *inFileName, ZvSettings *settings) :
	SyncIn(settings),
	zed_(NULL)
{
	zed_ = new Camera(inFileName);
	if (!zed_)
		return;

	InitParams parameters;
	parameters.mode = PERFORMANCE;
	parameters.unit = MILLIMETER;
	parameters.verbose = 1;
	// init computation mode of the zed
	ERRCODE err = zed_->init(parameters);

	// Quit if an error occurred
	if (err != SUCCESS)
	{
		cout << errcode2str(err) << endl;
		delete zed_;
		zed_ = NULL;
		return;
	}
	//only for Jetson K1/X1 - see if it helps
	Camera::sticktoCPUCore(2);

	width_  = zed_->getImageSize().width;
	height_ = zed_->getImageSize().height;

	params_.init(zed_, true);
	startThread();

	while (height_ > 700)
	{
		width_  /= 2;
		height_ /= 2;
	}
}

ZedSVOIn::~ZedSVOIn()
{
	stopThread();
	if (zed_)
		delete zed_;
}

bool ZedSVOIn::isOpened(void) const
{
	return zed_ != NULL; 
}


CameraParams ZedSVOIn::getCameraParams(void) const
{
	return params_.get();
}


bool ZedSVOIn::postLockUpdate(cv::Mat &frame, cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
	if (!zed_)
		return false;

	if (zed_->grab())
		return false;
	const bool left = true;
	sl::zed::Mat slFrame = zed_->retrieveImage(left ? SIDE::LEFT : SIDE::RIGHT);
	cvtColor(slMat2cvMat(slFrame), frame, CV_RGBA2RGB);

	sl::zed::Mat slDepth = zed_->retrieveMeasure(MEASURE::DEPTH);
	slMat2cvMat(slDepth).copyTo(depth);

	const float *pCloud = (const float *)zed_->retrieveMeasure(MEASURE::XYZRGBA).data;
	cloud.clear();
	for (int i = 0; i < (depth.rows * depth.cols); i++)
	{
		if (isValidMeasure(pCloud[i * 4]))
		{
			pcl::PointXYZRGB pt;
			pt.x = pCloud[i * 4 + 0];
			pt.y = pCloud[i * 4 + 1];
			pt.z = pCloud[i * 4 + 2];
			float color = pCloud[i * 4 + 3];
			// Color conversion (RGBA as float32 -> RGB as uint32)
			uint32_t color_uint = *(uint32_t*) &color;
			unsigned char* color_uchar = (unsigned char*) &color_uint;
			color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
			pt.rgb = *reinterpret_cast<float*> (&color_uint);
			cloud.push_back(pt);
		}
	}

	return true;
}


bool ZedSVOIn::postLockFrameNumber(int framenumber) 
{
	return zed_ && zed_->setSVOPosition(framenumber);
}


int ZedSVOIn::frameCount(void) const
{
	// Luckily getSVONumberOfFrames() returns -1 if we're
	// capturing from a camera, which is also what the rest
	// of our code expects in that case
	if (zed_)
		return zed_->getSVONumberOfFrames();

	// If using zms or a live camera, there's no way to tell
	return -1;
}


#else

ZedSVOIn::ZedSVOIn(const char *inFileName, ZvSettings *settings) :
	SyncIn(settings)
{
	(void)inFileName;
}


ZedSVOIn::~ZedSVOIn()
{
}


bool ZedSVOIn::postLockUpdate(cv::Mat &frame, cv::Mat &depth)
{
	(void)frame;
	(void)depth;
	return true;
}


bool ZedSVOIn::postLockFrameNumber(int framenumber) 
{
	(void)framenumber;
	return 0;
}
#endif
