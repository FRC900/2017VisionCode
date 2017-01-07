#pragma once

#include "opencv2_3_shim.hpp"
#include "objtype.hpp"
// Turn Window from a typedef into a class :
//   Private members are the rect, index from Window plus maybe a score?
//   Constructor takes Rect, size_t index
//   Maybe another constructor with Rect, size_t scaleIndex, float score
//   Need calls for 
//      - get score
//      - get rect
//      - rescaling rect by a fixed value - multiply or divide the x,y,
//        width, height by a passed-in constant
//      - get scaled rect - given a scaledImages array, return the rect
//        scaled back to fit correctly on the original image. Should be
//        something like double scale = scaledImages[index], 
//        return rect(scaled down by scale.  See the first for loop in runNMS
//        for an example of this
//      - get an image for the window. Pass in scaledImages and a Mat. 
//        Fill in the Mat with the image data pulled from the correct scaled
//        image (the one from the entry <index> in the scaledImage array).
//        See the top of the loop in runDetection for an example
//

struct NNDetectDebugInfo
{
	size_t initialWindows; // # of d12 sliding windows
	size_t d12In;          // # of windows passing depth test
	size_t d12DetectOut;   // # of windows passing d12 detectnet
	size_t d12NMSOut;      // # of windows passing d12 NMS == # input to d24
	size_t d24DetectOut;   // # of windows passing d24 detectnet
};


template <class MatT, class ClassifierT>
class NNDetect
{
	public:
		NNDetect(const std::vector<std::string> &d12Files,
			     const std::vector<std::string> &d24Files, 
	   		     const std::vector<std::string> &c12Files,
			     const std::vector<std::string> &c24Files, 
			     float hfov,
			     const ObjectType &objToDetect) :
			d12_(d12Files[0], d12Files[1], d12Files[2], d12Files[3], 192),
			d24_(d24Files[0], d24Files[1], d24Files[2], d24Files[3], 64),
			c12_(c12Files[0], c12Files[1], c12Files[2], c12Files[3], 64),
			c24_(c24Files[0], c24Files[1], c24Files[2], c24Files[3], 64),
			hfov_(hfov),
			objToDetect_(objToDetect)
		{
		}

		void detectMultiscale(const cv::Mat &inputImg,
				const cv::Mat &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				const double scaleFactor,
				const std::vector<double> &nmsThreshold,
				const std::vector<double> &detectThreshold,
				const std::vector<double> &calThreshold,
				std::vector<cv::Rect> &rectsOut,
				std::vector<cv::Rect> &uncalibRectsOut);

		bool initialized(void) const;

		NNDetectDebugInfo DebugInfo(void) const;
	private:
		typedef std::pair<cv::Rect, size_t> Window;
		ClassifierT d12_;
		ClassifierT d24_;
		ClassifierT c12_;
		ClassifierT c24_;
		float hfov_;
		ObjectType objToDetect_;
		NNDetectDebugInfo debug_;
		void doBatchPrediction(ClassifierT &classifier,
				const std::vector<MatT> &imgs,
				const float threshold,
				const std::string &label,
				std::vector<size_t> &detected,
				std::vector<float>  &scores);

		void generateInitialWindows(
				const MatT &input,
				const MatT &depthIn,
				const cv::Size &minSize,
				const cv::Size &maxSize,
				const int wsize,
				double scaleFactor,
				std::vector<std::pair<MatT, double> > &scaledimages,
				std::vector<Window> &windows);

		void runDetection(ClassifierT &classifier,
				const std::vector<std::pair<MatT, double> > &scaledimages,
				const std::vector<Window> &windows,
				const float threshold,
				const std::string &label,
				std::vector<Window> &windowsOut,
				std::vector<float> &scores);

		void runGlobalNMS(const std::vector<Window> &windows, 
				const std::vector<float> &scores,  
				const std::vector<std::pair<MatT, double> > &scaledImages,
				const double nmsThreshold,
				std::vector<Window> &windowsOut);
		void runLocalNMS(const std::vector<Window> &windows, 
				const std::vector<float> &scores,  
				const double nmsThreshold,
				std::vector<Window> &windowsOut);

		void runCalibration(const std::vector<Window>& windowsIn,
				    const std::vector<std::pair<MatT, double> > &scaledImages,
				    ClassifierT &classifier,
				    float threshold,
				    std::vector<Window>& windowsOut);

		void doBatchCalibration(ClassifierT &classifier,
					const std::vector<MatT>& imags,
					const float threshold,
					std::vector<std::vector<float> >& shift);

		void checkDepthList(const float depth_min, const float depth_max,
				const std::vector<cv::Mat> &depthList, std::vector<bool> &validList);
		void checkDepthList(const float depth_min, const float depth_max,
				const std::vector<GpuMat> &depthList, std::vector<bool> &validList);

		bool depthInRange(const float depth_min, const float depth_max, 
				const cv::Mat &detectCheck);
};
