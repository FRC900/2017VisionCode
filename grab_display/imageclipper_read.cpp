#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace boost::filesystem;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		if (!item.empty())
			elems.push_back(item);
	}
	return elems;
}

vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

// Using the data encoded in filename, read the frame
// the image was originally clipped from. Generate a bounding
// rect starting with the original image location, then
// adjust it to the correct aspect ratio.
// Return true if successful, false if an error happens
bool getFrameAndRect(const string &filename, const string &srcPath, const double AR,
		string &origName, int &frame, Mat &mat, Rect &rect)
{
	int rotation = 0;
	string name = filename.substr(0, filename.rfind('.'));
	vector<string> tokens = split(name, '_');
	frame       = atoi(tokens[tokens.size()-6].c_str());
	rect.x      = atoi(tokens[tokens.size()-5].c_str());
	rect.y      = atoi(tokens[tokens.size()-4].c_str());
	rect.width  = atoi(tokens[tokens.size()-3].c_str());
	rect.height = atoi(tokens[tokens.size()-2].c_str());

	origName = string();
	for (size_t i = 0; i < tokens.size() - 6; i++)
	{
		origName += tokens[i];
		if (i < tokens.size() - 7)
			origName += "_";
	}

	string fileExt = extension(origName);
	if ((fileExt == ".jpg") || (fileExt == ".JPG") || 
		(fileExt == ".png") || (fileExt == ".PNG") )
	{
		mat = imread((srcPath+origName).c_str());
	}
	else
	{

		VideoCapture cap((srcPath+origName).c_str());
		if( !cap.isOpened() )
		{
			cerr << "Can not open " << srcPath+origName << endl;
			return false;
		}
		cap.set(CV_CAP_PROP_POS_FRAMES, frame - 1);
		cap >> mat;
	}
	if (mat.empty())
	{
		cerr << "Can not read" << srcPath+origName << endl;
		return false;
	}

	const double ar = rect.width / (double)rect.height;
	float added_height = 0.0;
	float added_width  = 0.0;
	if (ar > AR)
	{
		added_height = rect.width / AR - rect.height;
		rect.x -= cvRound((added_height / 2.0) * sin(rotation / 180.0 * M_PI));
		rect.y -= cvRound((added_height / 2.0) * cos(rotation / 180.0 * M_PI));
	}
	else if (ar < AR)
	{
		added_width = rect.height * AR - rect.width;
		rect.x -= cvRound((added_width / 2.0) * cos(rotation / 180.0 * M_PI));
		rect.y += cvRound((added_width / 2.0) * sin(rotation / 180.0 * M_PI));
	}
	rect.width  += added_width;
	rect.height += added_height;

	if ((rect.x < 0) || (rect.y < 0) || 
		(rect.br().x >= mat.cols) || (rect.br().y >= mat.rows))
		return false;
	return true;
}


