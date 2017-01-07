#pragma once
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class RandomSubImage
{
	public:
		RandomSubImage(const cv::RNG &rng, const std::vector<std::string> &fileNames);

		cv::Mat get (double ar, double minPercent);

	private:
		cv::RNG rng_;
		std::vector<std::string> fileNames_;
		std::vector<std::shared_ptr<cv::Mat>> images_;
};



