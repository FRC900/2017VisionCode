#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include "zedsvoin.hpp"
#include "zmsin.hpp"
#include "zmsout.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		cout << argv[1] << " input output" << endl;
		return 0;
	}
	string ext = boost::filesystem::extension(argv[1]);
	MediaIn *in;

	if ((ext == ".svo") || (ext ==  ".SVO"))
		in = new ZedSVOIn(argv[1]);
	else if ((ext == ".zms") || (ext ==  ".ZMS"))
		in = new ZMSIn(argv[1]);
	else
	{
		cerr << "Unknown input file extension" << endl;
		return -1;
	}

	ZMSOut out(argv[2]);

	Mat image;
	Mat depth;
	while (in->getFrame(image, depth) )
	{
		out.sync();
		out.saveFrame(image, depth);
		cout << in->FPS() << " FPS" << endl;
	}
	out.sync();
	return 0;
}
