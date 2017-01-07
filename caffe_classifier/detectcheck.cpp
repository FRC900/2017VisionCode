#include <iostream>
#include <sstream>
#include "CaffeBatchPrediction.hpp"
#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#include "zedin.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;

static double gtod_wrapper(void)
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}

int main(int argc, char *argv[])
{
	if (argc != 10) 
	{
		cerr << "Usage: " << argv[0]
			<< " d12-deploy.prototxt d12-network.caffemodel"
			<< " d12-mean.binaryproto d12-labels.txt " 
			<< " d24-deploy.prototxt d24-network.caffemodel"
			<< " d24-mean.binaryproto d24-labels.txt img.jpg" << endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);
	vector<string> d12Info;
	vector<string> d24Info;
	d12Info.push_back(argv[1]);
	d12Info.push_back(argv[2]);
	d12Info.push_back(argv[3]);
	d12Info.push_back(argv[4]);
	d24Info.push_back(argv[5]);
	d24Info.push_back(argv[6]);
	d24Info.push_back(argv[7]);
	d24Info.push_back(argv[8]);
	Mat frame;
	Mat depthMat;
	ZedIn* cap;
	cap = new ZedIn(argv[9]);
	if(!cap->update() || !cap->getFrame(frame, depthMat))
	{
		cerr << "err" << endl;
		return 1;
	}
	NNDetect<cv::Mat> detect(d12Info, d24Info, 75. * M_PI/180.);
	Mat emptyMat;
	Size minSize(30,30);
	Size maxSize(700,700);
	vector<Rect> rectsOut;
	vector<Rect> depthRectsOut;
	vector<double> detectThresholds;
	detectThresholds.push_back(0.75);
	detectThresholds.push_back(0.5);
	vector<double> nmsThresholds;
	nmsThresholds.push_back(0.5);
	nmsThresholds.push_back(0.75);
	while(1)
	{
		cap->update();
		cap->getFrame(frame, depthMat);
		if(frame.empty())
		{
			break;
		}
		// min and max size of object we're looking for.  The input
		// image will be scaled so that these min and max sizes
		// line up with the classifier input size.  Other scales will
		// fill in the range between those two end points.
		detect.detectMultiscale(frame, emptyMat, minSize, maxSize, 1.15, nmsThresholds, detectThresholds, rectsOut);
		detect.detectMultiscale(frame, depthMat, minSize, maxSize, 1.15, nmsThresholds, detectThresholds, depthRectsOut);
		namedWindow("Image", WINDOW_AUTOSIZE);
		for (auto it = rectsOut.cbegin(); it != rectsOut.cend(); ++it)
		{
			cout << "TL: " << it->tl() << endl;
			cout << "Depth: " << depthMat.at<float>((it->tl().x+it->br().x)/2, (it->tl().y+it->br().y)/2) << endl;
			cout << "Allowable Mid: " << (192.9 * pow(((float)it->width*(float)it->height)/((float)cap->width()*(float)cap->height()), -.534)) << endl;
			rectangle(frame, *it, Scalar(0,0,255));
		}
		for (auto it = depthRectsOut.cbegin(); it != depthRectsOut.cend(); ++it)
		{
			cout << "Made it through!" << endl;
			rectangle(frame, *it, Scalar(255,0,0));
		}
		imshow("Image", frame);
		//imwrite("detect.png", inputImg);
		char c = waitKey(5);
		if(c == ' ')
		{
			waitKey(0);
		}
	}
}

