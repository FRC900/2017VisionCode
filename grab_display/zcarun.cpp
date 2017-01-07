// Code to apply ZCA transform to a list of input images
#include <fstream>
#include <string>
#include <sys/stat.h>
#include "zca.hpp"

#include "utilities_common.h"
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
using namespace cv::cuda;
#endif

static void doOutput(const vector<Mat> &outImgs, const vector<string> &filenames, const string &outdir)
{
	for (size_t i = 0; i < outImgs.size(); i++)
	{
		const size_t found = filenames[i].find_last_of("/\\");
		mkdir((outdir+"/"+filenames[i].substr(0,found)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		try
		{
			if (!imwrite(outdir+"/"+filenames[i], outImgs[i]))
				cerr << "Failure converting image to PNG format: " << outdir << "/" << filenames[i]<< endl;
		}
		catch (runtime_error& ex) 
		{
			cerr << "Exception converting image to PNG format: " << ex.what() << endl;
		}
	}
}

int main(int argc, char **argv)
{
	const int batchSize = 1024;
	if (argc <= 3)
	{
		cout << "Usage : " << argv[0] << " zcaWeightsFile filelist outdir" << endl;
		cout << "\tzcaWeights is a .xml or .zca file with the weights used to transform each image" << endl;
		cout << "\t\tIf in doubt use the one from the d12 or d24 subdirs in zebravision" << endl;
		cout << "\t filelist : a text file with one image name per line" << endl;
		cout << "\t outdir : directory to write the processed images into" << endl;
			
		return 1;
	}
	ZCA zca(argv[1], batchSize);

	Mat img; // full image data
	ifstream infile(argv[2]);
	string outdir = argv[3];

	string filename;
	int count = 0;

	mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	vector <Mat> imgs;
	vector <string> filenames;
	while (getline(infile, filename))
	{
		//cout << filename << endl;
		if ((++count % 10000) == 0)
			cout << count << endl;

		// Save a list of filenames and 
		// a list of Mats with the image
		filenames.push_back(filename);
		imgs.push_back(imread(filename));

		// Make sure the image was read correctly
		// if not, print it for debugging
		// Don't add an empty image to the list
		// to be processed, though.
		if (imgs[imgs.size()-1].empty())
		{
			cout << "\"" << filename <<  "\"" << endl;
			continue;
		}

		if (imgs.size() == batchSize)
		{
			//double start = gtod_wrapper();
			//
			// Actually apply the transformation
			// outImgs will be an array of mats, each
			// one being the processed version of
			// the corresponding index in the input imgs
			auto outImgs = zca.Transform8UC3(imgs);

			// double end = gtod_wrapper();
			//cout << "Elapsed time for ZCA : " << end - start << endl;

			// Write each output image to the output directory
			doOutput(outImgs, filenames, outdir);

			// Clear input arrays to start next batch
			// of data
			imgs.clear();
			filenames.clear();
		}
	}
	// Handle the final remaining images at the end
	// of the list.  This code will run when the list
	// length is not an exact multiple of the batch size.
	if (imgs.size())
	{
		auto outImgs = zca.Transform8UC3(imgs);
		doOutput(outImgs, filenames, outdir);
	}
	cout << zca.alpha() << " " << zca.beta() << endl;
}

