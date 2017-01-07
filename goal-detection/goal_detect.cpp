#include <iostream>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>

#include "zedcamerain.hpp"
#include "zedsvoin.hpp"
#include "zmsin.hpp"
#include "GoalDetector.hpp"
#include "Utilities.hpp"
#include "track3d.hpp"
#include "frameticker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	MediaIn *cap = NULL;
	if (argc == 2)
		cap = new ZMSIn(argv[1]);
	else
		cap = new ZedCameraIn(false);

	if (cap == NULL)
	{
		cerr << "Error creating input" << endl;
		return -1;
	}

	GoalDetector gd(Point2f(cap->getCameraParams().fov.x, 
				            cap->getCameraParams().fov.y), 
			        Size(cap->width(),cap->height()), true);

	zmq::context_t context(1);
	zmq::socket_t publisher(context, ZMQ_PUB);

	std::cout<< "Starting network publisher 5800" << std::endl;
	publisher.bind("tcp://*:5800");

	Mat image;
	Mat depth;
	//Mat depthNorm;
	Rect bound;
	FrameTicker frameTicker;
	while (cap->getFrame(image, depth))
	{
		frameTicker.mark();
		//imshow ("Normalized Depth", depthNorm);

		gd.processFrame(image, depth);
		gd.drawOnFrame(image);

		stringstream ss;
		ss << fixed << setprecision(2) << frameTicker.getFPS() << "FPS";
		putText(image, ss.str(), Point(image.cols - 15 * ss.str().length(), 50), FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255));
		rectangle(image, gd.goal_rect(), Scalar(255,0,0), 2);
		imshow ("Image", image);

		stringstream gString;
		gString << "G ";
		gString << fixed << setprecision(4) << gd.dist_to_goal() << " ";
		gString << fixed << setprecision(2) << gd.angle_to_goal();

		cout << "G : " << gString.str().length() << " : " << gString.str() << endl;
		zmq::message_t grequest(gString.str().length() - 1);
		memcpy((void *)grequest.data(), gString.str().c_str(), gString.str().length() - 1);
		publisher.send(grequest);

		if ((uchar)waitKey(5) == 27)
		{
			break;
		}
	}
	return 0;
}
