/**
 * Fast non-maximum suppression in C, port from  
 * http://quantombone.blogspot.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
 *
 * @blackball (bugway@gmail.com)
 */

#include <algorithm>
#include <functional>
#include <iostream>

#include "fast_nms.hpp"

using namespace std;
using namespace cv;

class DetectedPlusIndex
{
	public :
		DetectedPlusIndex(const Rect &rect, double score, size_t index) :
			rect_(rect),
			score_(score),
			index_(index),
			valid_(true)
	{}

		Rect   rect_;
		float  score_;
		size_t index_;
		bool   valid_;

		bool operator> (const DetectedPlusIndex &other) const
		{
			return score_ > other.score_;
		}
};

void fastNMS(const vector<Detected> &detected, double overlap_th, vector<size_t> &filteredList) 
{
	filteredList.clear(); // Clear out return array
	vector <DetectedPlusIndex> dpi;

	// Create a list that includes the detected input plus the
	// index into the list as it was passed in.  Keep the index
	// so we can pass the original unsorted index back to the caller.
	size_t idx = 0;
	for (auto it = detected.cbegin(); it != detected.end(); ++it)
		dpi.push_back(DetectedPlusIndex(it->first, it->second, idx++));

	// Sort input rects by decreasing score - i.e. look at best
	// values first
	sort(dpi.begin(), dpi.end(), greater<DetectedPlusIndex>());

	// Loop through the dpi array. Each time through, grab
	// the highest scoring remaining rect. Invalidate rects
	// which overlap and have lower scores. Repeat until
	// every rect has been the "best" or has been invalidated
	auto it = dpi.begin();
	auto dpi_end = dpi.end();
	do
	{
		// Look for the highest scoring unprocessed 
		// rectangle left in the list
		while ((it != dpi_end) && !it->valid_)
			++it;

		// Exit if everything has been processed
		if (it != dpi_end)
		{
			// Save the index of the highest ranked remaining Rect
			// and invalidate it - this means we've already
			// processed it
			filteredList.push_back(it->index_);
			it->valid_ = false;

			// Save this rect to compare against the
			// remaining lower-scoring ones
			Rect topRect = it->rect_;

			// Set up to continue processing next
			// rectangle next time through
			++it;

			// Loop through the rest of the array, looking
			// for entries which overlap with the current "good"
			// one being processed
			for (auto jt = it; jt != dpi_end; ++jt) 
			{
				// Only check entries which haven't 
				// been removed already
				if (jt->valid_)
				{
					Rect thisRect = jt->rect_;

					// Look at the Intersection over Union ratio.
					// The higher this is, the closer the two rects are
					// to overlapping
					double intersectArea = (topRect & thisRect).area();
					double unionArea     = topRect.area() + thisRect.area() - intersectArea;

					if ((intersectArea > 0.0) && ((1-(intersectArea / unionArea)) <= overlap_th))
						jt->valid_ = false; // invalidate Rects which overlap
				}
			}
		}
	}
	while (it != dpi_end);
}

#if 0
	static void 
test_nn() 
{
	vector<Detected> rects;
	vector<Rect> keep;

	rects.push_back(Detected(Rect(Point(0,  0),  Point(10+1, 10+1)), 0.5f));
	rects.push_back(Detected(Rect(Point(1,  1),  Point(10+1, 10+1)), 0.4f));
	rects.push_back(Detected(Rect(Point(20, 20), Point(40+1, 40+1)), 0.3f));
	rects.push_back(Detected(Rect(Point(20, 20), Point(40+1, 30+1)), 0.4f));
	rects.push_back(Detected(Rect(Point(15, 20), Point(40+1, 40+1)), 0.1f));

	fastNMS(rects, 0.4f, keep);

	for (size_t i = 0; i < keep.size(); i++)
		cout << keep[i] << endl;
}

	int 
main(int argc, char *argv[]) 
{
	test_nn();
	return 0;
}
#endif

