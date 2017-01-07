#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Point> FindObjPoints(const cv::Mat &frame, const cv::Scalar &rangeMin, const cv::Scalar &rangeMax);
bool FindRect(const cv::Mat& frame, const cv::Scalar &rangeMin, const cv::Scalar &rangeMax, cv::Rect& output);
bool getMask(const cv::Mat &frame, const cv::Scalar &rangeMin, const cv::Scalar &rangeMax, cv::Mat &objMask, cv::Rect &boundRect);

cv::Mat randomSubImage(cv::RNG &rng, const std::vector<std::string> &filenames, double ar, double minPercent);
cv::Mat doChromaKey(const cv::Mat &fgImage, const cv::Mat &bgImage, const cv::Mat &mask);
#if CV_MAJOR_VERSION == 2
cv::gpu::GpuMat doChromaKey(const cv::gpu::GpuMat &fgImage, const cv::gpu::GpuMat &bgImage, const cv::gpu::GpuMat &mask);
#elif CV_MAJOR_VERSION == 3
cv::cuda::GpuMat doChromaKey(const cv::cuda::GpuMat &fgImage, const cv::cuda::GpuMat &bgImage, const cv::cuda::GpuMat &mask);
#endif
