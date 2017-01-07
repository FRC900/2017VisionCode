#pragma once

#include <opencv2/opencv.hpp>

cv::Rect ResizeRect(const cv::Rect& rect, const cv::Size& size);
cv::Rect AdjustRect(const cv::Rect& rect, const double ratio);
bool RescaleRect(const cv::Rect& inRect, cv::Rect& outRect, const cv::Size& imageSize, const double scaleUp);

template <class MatT>
void rotateImageAndMask(const MatT &srcImg, const MatT &srcMask,
						const cv::Scalar &bgColor, const cv::Point3f &maxAngle,
						cv::RNG &rng, MatT &outImg, MatT &outMask);
