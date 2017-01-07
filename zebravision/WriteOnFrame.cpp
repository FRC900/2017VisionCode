#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <time.h>
#include "WriteOnFrame.hpp"

using namespace std;
using namespace cv;

WriteOnFrame::WriteOnFrame() {}

void WriteOnFrame::writeTime(cv::Mat &frame) { //write the time on the frame
	time_t rawTime;
	time(&rawTime);
	struct tm * localTime;
	localTime = localtime(&rawTime);
	char arrTime[100];
	strftime(arrTime, sizeof(arrTime), "%T %D", localTime);
	putText(frame,string(arrTime), Point(0,20), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(147,20,255), 1);
}

void WriteOnFrame::writeMatchNumTime(cv::Mat &frame, std::string matchNum, double matchTime) {

	string matchTimeString;
	matchNum = "Match Number:" + matchNum;
	matchTimeString = "Match Time: ";
	stringstream s;
	s << matchTimeString;
	s << matchTime;
	matchTimeString = s.str();
	putText(frame,matchNum,Point(0,40), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(147,20,255), 1);
	putText(frame,matchTimeString,Point(0,60), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(147,20,255), 1);
}

void WriteOnFrame::writeMatchNumTime(cv::Mat &frame) {
	string matchNum = "No Match Number";
	string matchTimeString = "No Match Time";
	putText(frame,matchNum,Point(0,40), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(147,20,255), 1);
	putText(frame,matchTimeString,Point(0,60), FONT_HERSHEY_TRIPLEX, 0.75, Scalar(147,20,255), 1);
}

#if 0
//this isn't fully implemented because I don't understand network tables

void WriteOnFrame::writeMatchNumTime(cv::mat &frame, NetworkTable netTable) {
	(void)netTable;
	string matchNum  = netTable->GetString("Match Number", "No Match Number");
	double matchTime = netTable->GetNumber("Match Time",-1);
	writeMatchNumTime(frame, matchNum, matchTime);
}

#endif
