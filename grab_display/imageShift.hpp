#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "random_subimage.hpp"

bool createShiftDirs(const std::string &outputDir);
void doShifts(const cv::Mat &src, const cv::Rect &objROI, cv::RNG &rng, const cv::Point3f &maxRot, int copiesPerShift, const std::string &outputDir, const std::string &fileName);
int doShifts(const cv::Mat &original, const cv::Mat &objMask, cv::RNG &rng, RandomSubImage &rsi, const cv::Point3f &maxRot, int copiesPerShift, const std::string &outputDir, const std::string &fileName);
