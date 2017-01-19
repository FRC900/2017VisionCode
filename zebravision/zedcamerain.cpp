#include <iostream>
#include "zedcamerain.hpp"
using namespace std;

#ifdef ZED_SUPPORT
#include <opencv2/imgproc/imgproc.hpp>

#include "cvMatSerialize.hpp"
#include "ZvSettings.hpp"

using namespace cv;
using namespace sl::zed;

void zedBrightnessCallback(int value, void *data);
void zedContrastCallback(int value, void *data);
void zedHueCallback(int value, void *data);
void zedSaturationCallback(int value, void *data);
void zedGainCallback(int value, void *data);
void zedExposureCallback(int value, void *data);

ZedCameraIn::ZedCameraIn(bool gui, ZvSettings *settings) :
	AsyncIn(settings),
	zed_(NULL),
	brightness_(2),
	contrast_(6),
	hue_(7),
	saturation_(4),
	gain_(1),
	exposure_(1) // Should set exposure = -1 => auto exposure/auto-gain
{
	if (!Camera::isZEDconnected()) // Open an actual camera for input
	{
		cerr << "ZED camera not found" << endl;
		return;
	}

	// Ball detection runs at ~10 FPS on Jetson
	// so run camera capture more slowly
#ifdef IS_JETSON
	zed_ = new Camera(HD720, 15);
#else
	zed_ = new Camera(HD720, 30);
#endif

	if (!zed_)
		return;

	InitParams parameters;
	parameters.mode = PERFORMANCE;
	parameters.unit = MILLIMETER;
	parameters.coordinate = RIGHT_HANDED; // OpenGL compatible
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

	width_  = zed_->getImageSize().width;
	height_ = zed_->getImageSize().height;

	if (!loadSettings())
		cerr << "Failed to load ZedCameraIn settings from XML" << endl;

	zedBrightnessCallback(brightness_, this);
	zedContrastCallback(contrast_, this);
	zedHueCallback(hue_, this);
	zedSaturationCallback(saturation_, this);
	zedGainCallback(gain_, this);
	zedExposureCallback(exposure_, this);

	cout << "brightness_ = " << zed_->getCameraSettingsValue(ZED_BRIGHTNESS) << endl;
	cout << "contrast_ = " << zed_->getCameraSettingsValue(ZED_CONTRAST) << endl;
	cout << "hue_ = " << zed_->getCameraSettingsValue(ZED_HUE) << endl;
	cout << "saturation_ = " << zed_->getCameraSettingsValue(ZED_SATURATION) << endl;
	cout << "gain_ = " << zed_->getCameraSettingsValue(ZED_GAIN) << endl;
	cout << "exposure_ = " << zed_->getCameraSettingsValue(ZED_EXPOSURE) << endl;
	if (gui)
	{
		cv::namedWindow("Adjustments", CV_WINDOW_NORMAL);
		cv::createTrackbar("Brightness", "Adjustments", &brightness_, 9, zedBrightnessCallback, this);
		cv::createTrackbar("Contrast", "Adjustments", &contrast_, 9, zedContrastCallback, this);
		cv::createTrackbar("Hue", "Adjustments", &hue_, 12, zedHueCallback, this);
		cv::createTrackbar("Saturation", "Adjustments", &saturation_, 9, zedSaturationCallback, this);
		cv::createTrackbar("Gain", "Adjustments", &gain_, 101, zedGainCallback, this);
		cv::createTrackbar("Exposure", "Adjustments", &exposure_, 102, zedExposureCallback, this);
	}

	while (height_ > 700)
	{
		width_  /= 2;
		height_ /= 2;
	}

	params_.init(zed_, true);
	startThread();
}


ZedCameraIn::~ZedCameraIn()
{
	if (!saveSettings())
		cerr << "Failed to save ZedCameraIn settings to XML" << endl;
	stopThread();
	if (zed_)
		delete zed_;
}


bool ZedCameraIn::loadSettings(void)
{
	if (settings_) {
		settings_->getInt(getClassName(), "brightness", brightness_);
		settings_->getInt(getClassName(), "contrast",   contrast_);
		settings_->getInt(getClassName(), "hue",        hue_);
		settings_->getInt(getClassName(), "saturation", saturation_);
		settings_->getInt(getClassName(), "gain",       gain_);
		settings_->getInt(getClassName(), "exposure",   exposure_);
		return true;
	}
	return false;
}


bool ZedCameraIn::saveSettings(void) const
{
	if (settings_) {
		settings_->setInt(getClassName(), "brightness", brightness_);
		settings_->setInt(getClassName(), "contrast",   contrast_);
		settings_->setInt(getClassName(), "hue",        hue_);
		settings_->setInt(getClassName(), "saturation", saturation_);
		settings_->setInt(getClassName(), "gain",       gain_);
		settings_->setInt(getClassName(), "exposure",   exposure_);
		settings_->save();
		return true;
	}
	return false;
}


bool ZedCameraIn::isOpened(void) const
{
	return zed_ ? true : false;
}


CameraParams ZedCameraIn::getCameraParams(void) const
{
	return params_.get();
}


// Code for grabbing data from camera into local buffers
// Doesn't need a lock since these aren't shared buffers
// with the main thread
bool ZedCameraIn::preLockUpdate(void)
{
	const bool left = true;
	int badReadCounter = 0;
	while (zed_->grab(SENSING_MODE::STANDARD))
	{
		// grab() will return true if the next
		// frame isn't ready yet
		boost::this_thread::interruption_point();
		// Wait a bit to see if the next
		// frame shows up
		usleep(5000);
		// Try to grab a bunch of times before
		// bailing out and failing
		if (++badReadCounter == 100)
			return false;
	}

	sl::zed::Mat slFrame = zed_->retrieveImage(left ? SIDE::LEFT : SIDE::RIGHT);
	cvtColor(slMat2cvMat(slFrame), localFrame_, CV_RGBA2RGB);

	sl::zed::Mat slDepth = zed_->retrieveMeasure(MEASURE::DEPTH);
	slMat2cvMat(slDepth).copyTo(localDepth_);

	const float *pCloud = (const float *)zed_->retrieveMeasure(MEASURE::XYZRGBA).data;
	localCloud_.clear();
	for (int i = 0; i < (localDepth_.rows * localDepth_.cols); i++)
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
			localCloud_.push_back(pt);
		}
	}

	return true;
}


bool ZedCameraIn::postLockUpdate(cv::Mat &frame, cv::Mat &depth, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
	localFrame_.copyTo(frame);
	localDepth_.copyTo(depth);
	cloud = localCloud_;

	return true;
}


void zedBrightnessCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->brightness_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_BRIGHTNESS, value - 1, value == 0);
}


void zedContrastCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->contrast_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_CONTRAST, value - 1, value == 0);
}


void zedHueCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->hue_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_HUE, value - 1, value == 0);
}


void zedSaturationCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->saturation_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_SATURATION, value - 1, value == 0);
}


void zedGainCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->gain_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_GAIN, value - 1, value == 0);
}

void zedExposureCallback(int value, void *data)
{
    ZedCameraIn *zedPtr = static_cast<ZedCameraIn *>(data);
	zedPtr->gain_ = value;
	if (zedPtr->zed_)
		zedPtr->zed_->setCameraSettingsValue(ZED_EXPOSURE, value - 2, value == 0);
}


#else

ZedCameraIn::ZedCameraIn(bool gui, ZvSettings *settings) :
	AsyncIn(settings)
{
	(void)gui;
}

ZedCameraIn::~ZedCameraIn()
{
}


bool ZedCameraIn::preLockUpdate(void)
{
	return true;
}


bool ZedCameraIn::postLockUpdate(cv::Mat &frame, cv::Mat &depth,pcl::PointCloud<pcl::PointXYZRGB> &cloud )
{
	(void)frame;
	(void)depth;
	return false;
}
#endif

