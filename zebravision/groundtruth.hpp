#ifndef INC__GROUNDTRUTH__HPP__
#define INC__GROUNDTRUTH__HPP__

#include <map>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

// Simple class to store ground truth data. A ground truth entry is just a 
// known location of the object we're detecting in a video. Here, they're 
// stored as a video name, frame number and location rectangle.
class GroundTruth
{
	public :
		GroundTruth(const std::string &truthFile, const std::string &videoFile);

		// Get the next frame which has gt information.
		// Return -1 if end of frame list is reached
		int nextFrameNumber(void);

		// Number of frames with GT data
		size_t frameCount(void) const;
		
		// Get a list of GT rectangles for the given frame
		std::vector<cv::Rect> get(unsigned int frame) const;

		// Search for hits and false positives in the given frame
		// Return list of rects which were correctly found
		void processFrame(int frameNum, const std::vector<cv::Rect> &detectRects, double overlap, std::vector<cv::Rect> &found, std::vector<cv::Rect> &missed);
		void print(void) const;

	private :
		// Map of frame_num to list of GT rects for that frame
		std::map< unsigned int, std::vector<cv::Rect> > map_;

		// List of valid frames with GT data
		std::vector <unsigned int> frameList_;

		// Currently processed frame from above
		size_t frameListIdx_;

		// GT stats
		unsigned count_;
		unsigned found_;
		unsigned falsePositives_;
		unsigned framesSeen_;
};

#endif
