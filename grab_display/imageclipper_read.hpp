#pragma once

#include <string>

#include <opencv2/opencv.hpp>

// Using the data encoded in filename, read the frame
// the image was originally clipped from. Generate a bounding
// rect starting with the original image location, then
// adjust it to the correct aspect ratio.
// Return true if successful, false if an error happens
bool getFrameAndRect(const std::string &filename, const std::string &srcPath,
		const double AR, 
		std::string &origName, int &frame, cv::Mat &mat, cv::Rect &rect);
