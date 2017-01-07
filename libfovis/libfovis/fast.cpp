#include <stdint.h>
#include "fast.hpp"

#include <opencv2/opencv.hpp>
using namespace cv;
#if CV_MAJOR_VERSION == 2
#include <opencv2/gpu/gpu.hpp>
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
#include <opencv2/core/cuda.hpp>
using namespace cv::cuda;
#endif

namespace fovis
{

	void FAST(uint8_t* image, int width, int height, int row_stride,
			std::vector<KeyPoint>* keypoints, int threshold, bool nonmax_suppression )
	{
		std::vector<cv::KeyPoint> cvKeypoints;
		Mat frameCPU(height,width,CV_8UC1,image);
		if (getCudaEnabledDeviceCount() > 0)
		{
			GpuMat frameGPU(height,width,CV_8UC1); //initialize mats

			frameGPU.upload(frameCPU); //copy frame from cpu to gpu

			GpuMat mask(height,width,CV_8UC1);
			mask.setTo(Scalar(255,255,255)); //create a mask to run the detection on the whole image

#if CV_MAJOR_VERSION == 2
			FAST_GPU FASTObject(threshold,nonmax_suppression); //run the detection
			FASTObject(frameGPU,mask,cvKeypoints);

#elif CV_MAJOR_VERSION == 3
			Ptr<cv::cuda::FastFeatureDetector> fast = cv::cuda::FastFeatureDetector::create(threshold, nonmax_suppression);
			fast->detect(frameGPU, cvKeypoints, mask);
#endif
		}
		else
		{
			FAST(frameCPU, cvKeypoints, threshold, nonmax_suppression);
		}

		keypoints->clear();
		for(uint i = 0; i < cvKeypoints.size(); i++) { //the pointers here are so ugly please
			keypoints->push_back(fovis::KeyPoint()); //don't read these lines
			(*keypoints)[i].u = cvKeypoints[i].pt.x; //just follow the comments
			(*keypoints)[i].v = cvKeypoints[i].pt.y; //because if you read them all the 
			(*keypoints)[i].score = cvKeypoints[i].response; //way you can skip reading this section of code
		}
	}

}
