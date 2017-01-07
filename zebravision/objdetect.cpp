#include <iostream>
#include <sys/stat.h>

#include "objdetect.hpp"
#if CV_MAJOR_VERSION == 2
#define cuda gpu
#elif CV_MAJOR_VERSION == 3
#include <opencv2/cudaimgproc.hpp>
#endif

int scale         =  10;
int d12NmsThreshold = 40;
int d24NmsThreshold = 98;
int minDetectSize = 44; // overridden in main()
int maxDetectSize = 750;
int d12Threshold  = 45; // overridden in main()
int d24Threshold  = 98; // overridden in main()
int c12Threshold  = 17; // overridden in main()
int c24Threshold  = 5; // overridden in main()
int neighbors = 4;

// TODO : make this a parameter to the detect code
// so that we can detect objects with different aspect ratios
const double DETECT_ASPECT_RATIO = 1.0;

using namespace std;
using namespace cv;

// Base class for object detection. Doesn't do much - derived 
// classes need to implement the important stuff
ObjDetect::ObjDetect(void) :
	init_(false)
{
}

ObjDetect::~ObjDetect()
{
}

std::vector<size_t> ObjDetect::DebugInfo(void) const
{
	return std::vector<size_t>();
}

bool ObjDetect::initialized(void) const
{
	return init_;
}

// OpenCV CPU CascadeClassifier
ObjDetectCascadeCPU::ObjDetectCascadeCPU(const std::string &cascadeName) :
	ObjDetect()
{ 
	struct stat statbuf;
	if (stat(cascadeName.c_str(), &statbuf) != 0)
	{
		std::cerr << "Can not open classifier input " << cascadeName << std::endl;
		std::cerr << "Try to point to a different one with --classifierBase= ?" << std::endl;
		return;
	}
	std::cout << "loading classifier " << cascadeName << std::endl;
	init_ = classifier_.load(cascadeName);
}

void ObjDetectCascadeCPU::Detect(const Mat &frame, 
								const Mat &depthIn, 
								vector<Rect> &imageRects, 
								vector<Rect> &uncalibImageRects)
{
	(void)depthIn;
	uncalibImageRects.clear();
	Mat frameGray;
	Mat frameEq;
	cvtColor(frame, frameGray, CV_BGR2GRAY);
	equalizeHist(frameGray, frameEq);

	classifier_.detectMultiScale(frameEq,
			imageRects,
			1.01 + scale/100.,
			neighbors,
			CV_HAAR_SCALE_IMAGE,
			Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
			Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize) );
}


// OpenCV GPU CascadeClassifier
ObjDetectCascadeGPU::ObjDetectCascadeGPU(const std::string &cascadeName) :
	ObjDetect()
{ 
	struct stat statbuf;
	if (stat(cascadeName.c_str(), &statbuf) != 0)
	{
		std::cerr << "Can not open classifier input " << cascadeName << std::endl;
		std::cerr << "Try to point to a different one with --classifierBase= ?" << std::endl;
		return;
	}
#if CV_MAJOR_VERSION == 2
	 init_ = classifier_.load(cascadeName);
#else
	classifier_ = cuda::CascadeClassifier::create(cascadeName);
	if (classifier_ != NULL)
		init_ = true;
#endif
}

void ObjDetectCascadeGPU::Detect(const Mat &frame, 
								const Mat &depthIn, 
								vector<Rect> &imageRects, 
								vector<Rect> &uncalibImageRects)
{
	(void)depthIn;
	uncalibImageRects.clear();
	GpuMat frameGPU(frame);
	GpuMat frameGray;
	GpuMat frameEq;
	GpuMat resultGPU;
	cuda::cvtColor(frameGPU, frameGray, CV_BGR2GRAY);
	cuda::equalizeHist(frameGray, frameEq);

#if CV_MAJOR_VERSION == 2
	int detectCount = classifier_.detectMultiScale(frameEq, 
					resultGPU,
					Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
					Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
					1.01 + scale/100., 
					neighbors); 
	Mat result;
	resultGPU.colRange(0, detectCount).download(result);

	imageRects.clear();
	Rect *rects = result.ptr<Rect>();
	for(int i = 0; i < detectCount; ++i)
		imageRects.push_back(rects[i]);
#else
	classifier_->setMinObjectSize(Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize));
	classifier_->setMaxObjectSize(Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize));
	classifier_->setScaleFactor(1.01 + scale/100.);
	classifier_->setMinNeighbors(neighbors);
	classifier_->detectMultiScale(frameEq, resultGPU);
	classifier_->convert(resultGPU, imageRects);
#endif
}

// Base class for NNet-based sliding window
// classifier. Can process sliding windows in either 
// MatT = Mat or GpuMat.  ClassifierT can be Mat or GpuMat
// based Caffe or eventually Gpu-based TensorRT
template <class MatT, class ClassifierT>
ObjDetectNNet<MatT, ClassifierT>::ObjDetectNNet(
		std::vector<std::string> &d12Files,
		std::vector<std::string> &d24Files,
		std::vector<std::string> &c12Files,
		std::vector<std::string> &c24Files,
		float hfov,
		const ObjectType &objToDetect) :
	ObjDetect(),
	classifier_(d12Files, d24Files, c12Files, c24Files, hfov, objToDetect )
{
	init_ = classifier_.initialized();
}


// Basic version of detection used for all derived
// neural-net based classes.  Detection code is the 
// same for all even though they use different types
// of detectors and classifiers (GPU vs. CPU, GIE vs. Caffe, etc)
template <class MatT, class ClassifierT>
void ObjDetectNNet<MatT, ClassifierT>::Detect(
		const Mat &frameInput, 
		const Mat &depthIn, 
		vector<Rect> &imageRects, 
		vector<Rect> &uncalibImageRects)
{
	// Control detect threshold via sliders.
	// Hack - set D24 to 0 to bypass running it
	vector<double> detectThreshold;
	detectThreshold.push_back(d12Threshold / 100.);
	detectThreshold.push_back(d24Threshold / 100.);

	vector<double> nmsThreshold;
	nmsThreshold.push_back(d12NmsThreshold/100.);
	nmsThreshold.push_back(d24NmsThreshold/100.);

	vector<double> calThreshold;
	calThreshold.push_back(c12Threshold/100.);
	calThreshold.push_back(c24Threshold/100.);

	classifier_.detectMultiscale(frameInput,
			depthIn,
			Size(minDetectSize * DETECT_ASPECT_RATIO, minDetectSize),
			Size(maxDetectSize * DETECT_ASPECT_RATIO, maxDetectSize),
			1.01 + scale/100.,
			nmsThreshold,
			detectThreshold,
			calThreshold,
			imageRects,
			uncalibImageRects);
}

template <class MatT, class ClassifierT>
vector<size_t> ObjDetectNNet<MatT, ClassifierT>::DebugInfo(void) const
{
	vector<size_t> ret;
	NNDetectDebugInfo debug = classifier_.DebugInfo();
	ret.push_back(debug.initialWindows);
	ret.push_back(debug.d12In);
	ret.push_back(debug.d12DetectOut);
	ret.push_back(debug.d12NMSOut);
	ret.push_back(debug.d24DetectOut);
	return ret;
}

#ifndef USE_GIE 
template class ObjDetectNNet<Mat, CaffeClassifier<Mat>>;
template class ObjDetectNNet<GpuMat, CaffeClassifier<GpuMat>>;
#else
template class ObjDetectNNet<Mat, GIEClassifier<Mat>>;
template class ObjDetectNNet<GpuMat, GIEClassifier<GpuMat>>;
#endif
