#include <algorithm>
#include <fstream>
#include <iostream>

#include "groundtruth.hpp"

using namespace std;
using namespace cv;

// Simple class to store ground truth data. A ground truth entry is just a 
// known location of the object we're detecting in a video. Here, they're 
// stored as a video name, frame number and location rectangle.
//
// Constructor - read from the file on disk truthFile, grabbing only
// ground truth information for the input video videoFile.
GroundTruth::GroundTruth(const string &truthFile, const string &videoFile) :
	frameListIdx_(0),
	count_(0),
	found_(0),
	falsePositives_(0),
	framesSeen_(0)
{
	ifstream ifs(truthFile, ifstream::in);
	Rect rect;
	int frame;
	string videoName;

	while (ifs >> videoName >> frame >> rect.x >> rect.y >> rect.width >> rect.height)
	{
		if (videoName == videoFile.substr(videoFile.find_last_of("\\/") + 1))
		{
			// Creates an entry if not there.  If one is already
			// there, just reuse the vector as is - this allows
			// for the possibility of mulitple rects per frame
			// if multiple copies of the object are present
			map_[frame].push_back(rect);
		}
	}

	// Generate a list of frame numbers which have valid
	// ground truth data - use that to skip ahead to the 
	// next GT frame to process
	for (auto it = map_.cbegin(); it != map_.cend(); ++it)
		frameList_.push_back(it->first);

	sort(frameList_.begin(), frameList_.end());
}


// Grab the list of ground truths for a given frame
vector<Rect> GroundTruth::get(unsigned int frame) const
{
	auto it = map_.find(frame);
	if (it == map_.end())
	{
		return vector<Rect>();
	}

	return it->second;
}

// How many frames have ground truth data?
size_t GroundTruth::frameCount(void) const
{
	return frameList_.size();
}

// Get the next fram with GT data
int GroundTruth::nextFrameNumber(void)
{
	if (frameListIdx_ < frameList_.size())
		return frameList_[frameListIdx_++];
	return -1;
}

// Process a frame. Update number of GTs actually in
// the frame, number detected and number of false
// positives found
vector<Rect> GroundTruth::processFrame(int frameNum, const vector<Rect> &detectRects)
{
	const vector<Rect> &groundTruthList = get(frameNum);
	vector<bool> groundTruthsHit(groundTruthList.size());
	vector<bool> detectRectsUsed(detectRects.size());
	vector<Rect> retList;

	count_ += groundTruthList.size();
	for(auto gt = groundTruthList.cbegin(); gt != groundTruthList.cend(); ++gt)
	{
		for(auto it = detectRects.cbegin(); it != detectRects.cend(); ++it)
		{
			// If the intersection is > 45% of the area of
			// the ground truth, that's a success
			if ((*it & *gt).area() > (max(gt->area(), it->area()) * 0.45))
			{
				if (!groundTruthsHit[gt - groundTruthList.begin()])
				{
					found_ += 1;
					groundTruthsHit[gt - groundTruthList.begin()] = true;
				}
				detectRectsUsed[it - detectRects.begin()] = true;
				retList.push_back(*it);
			}
		}
	}
	for(auto it = detectRectsUsed.cbegin(); it != detectRectsUsed.cend(); ++it)
		if (!*it)
			falsePositives_ += 1;

	framesSeen_ += 1;
	return retList;
}

// Print a summary of the results so far
void GroundTruth::print(void) const
{
	if (count_)
	{
		cout << found_ << " of " << count_ << " ground truth objects found (" << (double)found_ / count_ * 100.0 << "%)" << endl;
		cout << falsePositives_ << " false positives found in " << framesSeen_ << " frames (" << (double)falsePositives_/framesSeen_ << " per frame)" << endl;
	}
}
