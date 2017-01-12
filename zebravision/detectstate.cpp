// This class controls which detector is created
// and used.  It holds the current config files
// for the detector as well as various flags 
// controlling detector state (GPU vs CPU, etc)
//
// Methods are provided to change these settings
// Each frame the code runs update(). If any settings
// have changed since the last frame, the old
// detector is deleted and a new one is created
// with the updated settings.
#include <iostream>
#include <string>

#include "detectstate.hpp"

using namespace std;
using namespace cv;

#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
using namespace cv::cuda;
#endif

// Classifier IO holds the directory which has
// net description, weights, labels, etc.
// It also stores the index of the weight file
// to use - each epoch is saved and we can switch
// between them if needed
// hfov should be removed once the detect call takes
// a tracked object as input
// the flags control which parts of the detector
// use GPU vs CPU code
DetectState::DetectState(const ClassifierIO &d12IO, 
		const ClassifierIO &d24IO,
	   	const ClassifierIO &c12IO, 
		const ClassifierIO &c24IO, 
		float hfov, 
		bool gpu,
	   	bool tensorRT,
		const ObjectType &objToDetect) :
    detector_(NULL),
	d12IO_(d12IO),
	d24IO_(d24IO),
	c12IO_(c12IO),
	c24IO_(c24IO),
	casIO_("/home/ubuntu/2017VisionCode/zebravision/classifier_bin_", 14, 29),
	objToDetect_(objToDetect),
	hfov_(hfov),
	gpu_(gpu),
	tensorRT_(tensorRT),
	cascade_(false),
	oldGpu_(gpu),
	oldTensorRT_(tensorRT),
	oldCascade_(false),
	reload_(true)
{
	update();
}

DetectState::~DetectState()
{
	if (detector_)
		delete detector_;
}

// Grab file names needed to load a given classifier
// Check that the results make sense and return
// them to the caller. 
bool DetectState::checkNNetFiles(const ClassifierIO &inCLIO,
								 const string &name,
								 vector<string> &outFiles)
{
    outFiles = inCLIO.getClassifierFiles();
    for (size_t i = 0; i < outFiles.size(); ++i)
    {
        cerr << name << "[" << i << "] = " << outFiles[i] << endl;
    }
    if (outFiles.size() != 4)
    {
        cerr << "Wrong number of " << name << " to load classifier" << endl;
        return false;
    }
	return true;
}

// Called each frame. Reloads the detector if
// any settings have changed
bool DetectState::update(void)
{
	if (reload_ == false)
		return true;

	vector<string> d12Files;
	vector<string> d24Files;
	vector<string> c12Files;
	vector<string> c24Files;

	if (!cascade_)
		if (!checkNNetFiles(d12IO_, "D12Files", d12Files) ||
			!checkNNetFiles(d24IO_, "C24Files", d24Files) ||
			!checkNNetFiles(c12IO_, "D12Files", c12Files) ||
			!checkNNetFiles(c24IO_, "C24Files", c24Files))
			return false;

	// Save old detector state in case a problem
	// occurs
	ObjDetect *oldDetector = detector_;

	if (!cascade_)
	{
#ifndef USE_TensorRT
		//if (!tensorRT_)
		{
			if (!gpu_)
				detector_ = new ObjDetectCaffeCPU(d12Files, d24Files, c12Files, c24Files, hfov_, objToDetect_);
			else
				detector_ = new ObjDetectCaffeGPU(d12Files, d24Files, c12Files, c24Files, hfov_, objToDetect_);
		}
#else
		//else
		{
			if (!gpu_)
				detector_ = new ObjDetectTensorRTCPU(d12Files, d24Files, c12Files, c24Files, hfov_, objToDetect_);
			else
				detector_ = new ObjDetectTensorRTGPU(d12Files, d24Files, c12Files, c24Files, hfov_, objToDetect_);
		}
#endif
	}
	else
	{
		string name = casIO_.getClassifierName();
		if (!gpu_)
			detector_ = new ObjDetectCascadeCPU(name);
		else
			detector_ = new ObjDetectCascadeGPU(name);
	}

	reload_ = false;

	// Verfiy the load
	if( !detector_ || !detector_->initialized() )
	{
		cerr << "Error loading detector" << endl;
		detector_ = oldDetector;
		gpu_ = oldGpu_;
		tensorRT_ = oldTensorRT_;
		cascade_ = oldCascade_;
		return (oldDetector != NULL);
	}

	if (oldDetector)
		delete oldDetector;
	oldGpu_ = gpu_;
	oldTensorRT_ = tensorRT_;
	oldCascade_ = cascade_;

	return true;
}

void DetectState::toggleGPU(void)
{
	if (getCudaEnabledDeviceCount() > 0)
	{
		gpu_ = !gpu_;
		reload_ = true;
	}
}

void DetectState::toggleTensorRT(void)
{
	if (getCudaEnabledDeviceCount() > 0)
	{
		tensorRT_ = !tensorRT_;
		reload_ = true;
	}
}


void DetectState::toggleCascade(void)
{
	cascade_ = !cascade_;
	reload_ = true;
}

void DetectState::changeD12SubModel(bool increment)
{
	if (cascade_)
	{
		if (casIO_.findNextClassifierStage(increment))
			reload_ = true;
	}
	else if (d12IO_.findNextClassifierStage(increment))
		reload_ = true;
}

void DetectState::changeD12Model(bool increment)
{
	if (cascade_)
	{
		if (casIO_.findNextClassifierDir(increment))
			reload_ = true;
	}
	else 
		if (d12IO_.findNextClassifierDir(increment))
			reload_ = true;
}

void DetectState::changeD24SubModel(bool increment)
{
	if (!cascade_ && d24IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeD24Model(bool increment)
{
   if (!cascade_ && d24IO_.findNextClassifierDir(increment))
	  reload_ = true;
}

void DetectState::changeC12SubModel(bool increment)
{
   if (!cascade_ && c12IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeC12Model(bool increment)
{
   if (!cascade_ && c12IO_.findNextClassifierDir(increment))
	  reload_ = true;
}

void DetectState::changeC24SubModel(bool increment)
{
   if (!cascade_ && c24IO_.findNextClassifierStage(increment))
	  reload_ = true;
}

void DetectState::changeC24Model(bool increment)
{
	if (!cascade_ && c24IO_.findNextClassifierDir(increment))
		reload_ = true;
}

std::string DetectState::print(void) const
{
	string ret;
	if (gpu_)
		ret += "GPU_";
	else
		ret += "CPU_";
	if (cascade_)
	{
		ret += "Cascade";
		ret += casIO_.print();
	}
	else
	{
		if (tensorRT_)
			ret += "TensorRT";
		else
			ret += "Caffe";
		ret += " " + d12IO_.print() + "," + d24IO_.print() + "," + c12IO_.print() + "," + c24IO_.print();
	}
	return ret;
}
