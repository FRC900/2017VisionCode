#include "random_subimage.hpp"

using namespace std;
using namespace cv;

// Class used to grab a random subimage from a randomly selected image
// from a list passed in to the constructor
// Used to generate background images to superimpose images onto

RandomSubImage::RandomSubImage(const RNG &rng, const vector<string> &fileNames) :
rng_(rng),
	fileNames_(fileNames)
{
	images_.resize(fileNames_.size());
}

Mat RandomSubImage::get(double ar, double minPercent)
{
	// If no images are provided, generate a random background
	if (fileNames_.size() == 0)
	{
		Mat mat(320, 320/ar, CV_8UC3);
		rng_.fill(mat, RNG::UNIFORM, 0, 256);
		return mat;
	}
	while(1)
	{
		// Grab a random image from the list
		size_t idx = rng_.uniform(0, fileNames_.size());

		// Load it if necessary, otherwise just
		// re-use previously loaded copy
		if (images_[idx] == NULL)
		{
			images_[idx] = make_shared<Mat>(imread(fileNames_[idx]));
		}

		if ((images_[idx] == NULL) || images_[idx]->empty())
		{
			cerr << "Could not open background image " << fileNames_[idx] << endl;
			continue;
		}

		// Grab a percentage of the original image
		// with the requested aspect ratio
		double percent = rng_.uniform(minPercent, 1.0);
		Point2f pt(images_[idx]->cols * percent, 
			       images_[idx]->cols * percent / ar);

		// If the selected window ends up off the
		// edge of the image, scale it back down to fit
		if (cvRound(pt.y) > images_[idx]->rows)
		{
			pt.x = images_[idx]->rows * ar;
			pt.y = images_[idx]->rows;
		}

		// Round to integer sizes
		Size size (cvRound(pt.x), cvRound(pt.y));

		// Pick a random starting row and column from the image
		// Make sure the sub-image fits in the original
		// image
		Point tl(rng_.uniform(0, images_[idx]->cols - size.width),
			 	 rng_.uniform(0, images_[idx]->rows - size.height));

		return (*images_[idx])(Rect(tl, size));
	}
}

