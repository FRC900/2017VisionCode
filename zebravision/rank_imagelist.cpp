#include "CaffeClassifier.hpp"
#include "classifierio.hpp"

using namespace std;
using namespace cv;
#if CV_MAJOR_VERSION == 2
using namespace cv::gpu;
#elif CV_MAJOR_VERSION == 3
using namespace cv::cuda;
#endif

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		cout << "Usage : " << argv[0] << " filelist_of_imgs.txt" << endl;
		return 1;
	}
	ClassifierIO clio("d12", -1, -1);
	vector<string> files = clio.getClassifierFiles();
	for (auto it = files.begin(); it != files.end(); ++it)
	{
		*it = "/home/ubuntu/2016VisionCode/zebravision/" + *it;
		cout << *it << endl;
	}

	ifstream infile(argv[1]);

	CaffeClassifier<GpuMat> c(files[0], files[1], files[2], files[3], 256); 

	Mat img;
	Mat rsz;
	Mat f32;
	vector<GpuMat> imgs;
	vector<string> filenames;
	vector<pair<string, double>> predictions;
	string filename;
	while(getline(infile, filename))
	{
		//cout << "Read " << filename << endl;
		filenames.push_back(filename);
		img = imread(filename);
		cv::resize(img, rsz, c.getInputGeometry()); 
		rsz.convertTo(f32, CV_32FC3);
		imgs.push_back(GpuMat(f32).clone());
		if (imgs.size() == c.batchSize())
		{
			auto p = c.ClassifyBatch(imgs,2);
			for (auto v = p.cbegin(); v != p.cend(); ++v)
			{
				for (auto it = v->cbegin(); it != v->cend(); ++it)
				{
					if (it->first == "ball")
					{
						predictions.push_back(make_pair(filenames[v-p.cbegin()], it->second));
						//cout << filenames[v-p.cbegin()] << " " << it->second << endl;
					}
				}
			}
			filenames.clear();
			imgs.clear();
		}
	}
	if (imgs.size())
	{
		auto p = c.ClassifyBatch(imgs,2);
		for (auto v = p.cbegin(); v != p.cend(); ++v)
		{
			for (auto it = v->cbegin(); it != v->cend(); ++it)
			{
				if (it->first == "ball")
				{
					predictions.push_back(make_pair(filenames[v-p.cbegin()], it->second));
					//cout << filenames[v-p.cbegin()] << " " << it->second << endl;
				}
			}
		}
		filenames.clear();
		imgs.clear();
	}
	sort(predictions.begin(), predictions.end(), 
			[](const pair<string,double> &left, const pair<string,double> &right) 
			{
			return left.second < right.second;
			});
	for (auto it = predictions.cbegin(); it != predictions.cend(); ++it)
		cout << it->first << " " << it->second << endl;
	return 0;

}
