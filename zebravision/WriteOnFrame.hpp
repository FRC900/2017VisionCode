#ifndef INC_WRITEONFRAME_HPP__
#define INC_WRITEONFRAME_HPP__

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>

class WriteOnFrame {
	public:
		WriteOnFrame();
		void writeTime(cv::Mat &frame);
		void writeMatchNumTime(cv::Mat &frame, std::string matchNum, double matchTime);
		void writeMatchNumTime(cv::Mat &frame);
	private:
		std::string _matchNum;
};
#endif
