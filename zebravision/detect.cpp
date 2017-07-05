#include <iostream>
#include <sys/time.h>
#include "opencv2_3_shim.hpp"

#include "scalefactor.hpp"
#include "fast_nms.hpp"
#include "detect.hpp"
#ifndef USE_GIE
#include "CaffeClassifier.hpp"
#endif
#include "GIEClassifier.hpp"
#include "Utilities.hpp"


//#define VERBOSE

using namespace std;
using namespace cv;

#if 0
static double gtod_wrapper(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif


// Simple multi-scale detect.  Take a single image, scale it into a number
// of diffent sized images. Run a fixed-size detection window across each
// of them.  Keep track of the scale of each scaled image to map the
// detected rectangles back to the correct location and size on the
// original input images
// This code uses a cascade of neural nets to detect objects. The first
// neural net used is a simple, quick one. This quickly eliminates easy
// to reject objects but leaves many false positives. The second level
// net is larger and slower but also more accurate.  
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::detectMultiscale(const Mat&            inputImg,
												   const Mat&            depthMat,
												   const Size&           minSize,
												   const Size&           maxSize,
												   const double          scaleFactor,
												   const vector<double>& nmsThreshold,
												   const vector<double>& detectThreshold,
												   const vector<double>& calibrationThreshold,
												   vector<Rect>&         rectsOut,
												   vector<Rect>&         uncalibRectsOut)
{
    // Size of the first level classifier. Others are an integer multiple
    // of this initial size (2x and maybe 4x if we need it)
    const int wsize = d12_.getInputGeometry().width;

	// The neural nets take a fixed size input.  To detect various
	// different object sizes, pass in several different resizings
	// of the input image.  These vectors hold those resized images
	// plus the scale factor to return objects detected in them
	// to the correct size on the original input image
    vector<pair<MatT, double> > scaledImages12;
    vector<pair<MatT, double> > scaledImages24;
    // Maybe later ? vector<pair<MatT, double> > scaledImages48;

    // list of windows to work with.
    // A window is a rectangle from a given scaled image along
    // with the index of the scaled image it corresponds with.
	// Keeping both allows the code to resize the window to 
	// the correct size and location on the original input image
    vector<Window> windowsIn;
    vector<Window> windowsMid;
    vector<Window> windowsOut;
    vector<Window> uncalibWindowsOut;

    // Confidence scores (0.0 - 1.0) for each detected rectangle
	// Higher confidence means more likely to be what the net is 
	// looking for
    vector<float> scores;

    // Generate a list of initial windows to search. Each window will be a 12x12 image from 
	// a scaled copy of the full input image. These scaled images let us search for 
	// variable sized objects using a fixed-width detector
    MatT f32Img;

	// classifier runs on float pixel data. Convert it once here
	// rather than every time we pass a sub-window into the detection
	// code to save some time
    MatT(inputImg).convertTo(f32Img, CV_32FC3);

	// For GPU Mat, upload input CPU depth mat to GPU mat
	MatT depth(depthMat);
    generateInitialWindows(f32Img, depth, minSize, maxSize, wsize, scaleFactor, scaledImages12, windowsIn);

    // Generate scaled images for the larger net sizes as well.  Using a separate
	// set of scaled images for the 24x24 net will allow the code to grab
	// the images for those at greater detail rather than just resizing
	// a 12x12 image up to 24x24
    scalefactor(f32Img, scaledImages12, 2, scaledImages24);
    //scalefactor(f32Img, scaledImages12, 4, scaledImages48);

    // Do 1st level of detection. This takes the initial list of windows
    // and returns the list which have a score for "ball" above the
    // threshold listed.
    runDetection(d12_, scaledImages12, windowsIn, detectThreshold[0], objToDetect_.name(), windowsMid, scores);
	debug_.d12DetectOut = windowsMid.size();
    if ((detectThreshold.size() == 1) || (detectThreshold[1] <= 0.0))
	{
		runLocalNMS(windowsMid, scores, nmsThreshold[0], uncalibWindowsOut);
	}
    runCalibration(windowsMid, scaledImages12, c12_, calibrationThreshold[0], windowsOut);
	// If not running d24/c24, use the d12 output as the
	// uncalibrated results
    runLocalNMS(windowsOut, scores, nmsThreshold[0], windowsIn);
	debug_.d12NMSOut = windowsIn.size();

    // Double the size of the rects to get from a 12x12 to 24x24
    // detection window.  Use scaledImages24 for the detection call
    // since that has the scales appropriate for the 24x24 detector
    for (auto it = windowsIn.begin(); it != windowsIn.end(); ++it)
    {
        it->first = Rect(it->first.x * 2, it->first.y * 2,
                         it->first.width * 2, it->first.height * 2);
    }

    if ((detectThreshold.size() > 1) && (detectThreshold[1] > 0.0))
    {
        //cout << "d24 windows in = " << windowsIn.size() << endl;
        runDetection(d24_, scaledImages24, windowsIn, detectThreshold[1], "ball", windowsMid, scores);
		debug_.d24DetectOut = windowsMid.size();
		// Save uncalibrated results for debugging
		runGlobalNMS(windowsMid, scores, scaledImages24, nmsThreshold[1], uncalibWindowsOut);
		// Use calibration nets to try and better align the 
		// detection rectangle
        runCalibration(windowsMid, scaledImages24, c24_, calibrationThreshold[1], windowsOut);
        runGlobalNMS(windowsOut, scores, scaledImages24, nmsThreshold[1], windowsIn);
    }

    // Final result - scale the output rectangles back to the
    // correct scale for the original sized image
    rectsOut.clear();
    for (auto it = windowsIn.cbegin(); it != windowsIn.cend(); ++it)
    {
        const double scale = scaledImages24[it->second].second;
        const Rect rect(it->first);
        const Rect scaledRect(Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
        rectsOut.push_back(scaledRect);
    }

	// Also return the uncalibrated results for debuging
    uncalibRectsOut.clear();
    for (auto it = uncalibWindowsOut.cbegin(); it != uncalibWindowsOut.cend(); ++it)
    {
		double scale;
		if ((detectThreshold.size() == 1) || (detectThreshold[1] <= 0.0))
			scale = scaledImages12[it->second].second;
		else
			scale = scaledImages24[it->second].second;
        const Rect rect(it->first);
        const Rect scaledRect(Rect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale));
        uncalibRectsOut.push_back(scaledRect);
    }
}


// Non-maximum suppression eliminates overlapping detection
// rectangles.  It looks for sets of overlapping rectangles
// (with the amount of overlap to tolerate as nmsThreshold)
// and keeps only the one with the best score, rejecting 
// the others.
// This method runs NMS across windows of all scales - that
// is, a window from one scale can overlap and eliminate
// a window at a different scale. Use this for the last
// level of detection only
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::runGlobalNMS(const vector<Window>& windows,
                                  const vector<float>& scores,
                                  const vector<pair<MatT, double> >& scaledImages,
                                  const double nmsThreshold,
                                  vector<Window>& windowsOut)
{
    if ((nmsThreshold > 0.0) && (nmsThreshold <= 1.0))
    {
        // Detected is a rect, score pair.
        vector<Detected> detected;

        // Need to scale each rect to the correct mapping to the
        // original image, since rectangles from multiple different
        // scales might overlap
        for (size_t i = 0; i < windows.size(); i++)
        {
            const double scale = scaledImages[windows[i].second].second;
            const Rect   rect(windows[i].first);
            const Rect   scaledRect(rect.x / scale, rect.y / scale, rect.width / scale, rect.height / scale);
            detected.push_back(Detected(scaledRect, scores[i]));
        }

        vector<size_t> nmsOut;
        fastNMS(detected, nmsThreshold, nmsOut);
        // Each entry of nmsOut is the index of a saved rect/scales
        // pair.  Save the entries from those indexes as the output
        windowsOut.clear();
        for (auto it = nmsOut.cbegin(); it != nmsOut.cend(); ++it)
        {
            windowsOut.push_back(windows[*it]);
        }
    }
    else
    {
        // If not running NMS, output is the same as the input
        windowsOut = windows;
    }
}

// Run NMS locally - only eliminate overlapping
// windows of the same scale. Leave potential overlaps
// from different scales alone
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::runLocalNMS(const vector<Window>& windows,
                                 const vector<float>& scores,
                                 const double nmsThreshold,
                                 vector<Window>& windowsOut)
{
    if ((nmsThreshold > 0.0) && (nmsThreshold <= 1.0))
    {
		// Find the max scale index in the windows array
		size_t maxScale = numeric_limits<size_t>::min();
        for (size_t i = 0; i < windows.size(); i++)
			maxScale = max(maxScale, windows[i].second);

        // Detected is a rect, score pair.
		// Create a separate array for each scale 
		// so they can be processed individually.
		// detected[x] is the list of detections
		// for scale x
        vector<vector<Detected> > detected(maxScale + 1);
		
		// Don't rescale windows since we're only comparing
		// against other windows from the same scale
		// Scaling them each by the same amount won't make
		// any difference and just wastes time
        for (size_t i = 0; i < windows.size(); i++)
		{
			size_t scaleIdx = windows[i].second;
            detected[scaleIdx].push_back(Detected(windows[i].first, scores[i]));
        }

		// Run NMS separately for each scale. Accumulate
		// results from all of the passing windows into 
		// windowsOut
        windowsOut.clear();
		for (size_t i = 0; i < detected.size(); i++)
		{
			vector<size_t> nmsOut;
			fastNMS(detected[i], nmsThreshold, nmsOut);
			// Each entry of nmsOut is the index of a saved rect/scales
			// pair.  Save the entries from those indexes as the output
			for (auto it = nmsOut.cbegin(); it != nmsOut.cend(); ++it)
			{
				// Window is a <rect, scale index> pair. The rect
				// is the first entry in the detected list, the
				// scale index is i.
				windowsOut.push_back(make_pair(detected[i][*it].first, i));
			}
		}
    }
    else
    {
        // If not running NMS, output is the same as the input
        windowsOut = windows;
    }
}


// Generate a set of scaled images.  The first scale resizes
// the image such that objects which were minSize are now 
// wsize - the size of the input to the neural net.  This allows
// the net to detect those. It then repeatedly creates new scaled
// images by resizing them by factor scaleFactor until the detection
// size is maxsize.  This allows the code to detect objects
// at multiple scales using a fixed-size neural net input
// For each of these scaled images, move a fixed wsize
// window across all the rows and columns.  This sets up
// the code to detect objects at multiple locations in each
// scaled image.
// If depth info is present, filter out windows which would
// give objects which are the wrong size (calculated from
// scale info and depth information). That is, for a given 
// depth we know how big in pixels the boulder should be.  If
// the scaled wsize value is too far away from that predicted
// size, anyting we detect in that rect can't really be a boulder
// and should be filtered out of the initial list of possible
// detection windows.
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::generateInitialWindows(
    const MatT& input,
    const MatT& depthIn,
    const Size& minSize,
    const Size& maxSize,
    const int wsize,
    const double scaleFactor,
    vector<pair<MatT, double> >& scaledImages,
    vector<Window>& windows)
{
    windows.clear();
    debug_.initialWindows = 0;

    // How many pixels to move the window for each step
    // We use 4 - the calibration step can adjust +/- 2 pixels
    // in each direction, which means they will correct for
    // anything which is actually centered in one of the
    // pixels we step over.
    const int step = 4;

    // Create array of scaled images for RGB 
	// and depth data
    scalefactor(input, Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledImages);
    vector<pair<MatT, double> > scaledDepth;
    if (!depthIn.empty())
    {
        scalefactor(depthIn, Size(wsize, wsize), minSize, maxSize, scaleFactor, scaledDepth);
    }

    // Main loop.  Look at each scaled image in turn
    for (size_t scale = 0; scale < scaledImages.size(); ++scale)
    {
        const float depth_multiplier = 0.2;
		const float depth_avg = objToDetect_.expectedDepth(Rect(0, 0, wsize, wsize), 
				                                           scaledImages[scale].first.size(), hfov_);

        float depth_min = depth_avg - depth_avg * depth_multiplier;
        float depth_max = depth_avg + depth_avg * depth_multiplier;
#if 0
        cout << fixed << "Target size:" << wsize / scaledImages[scale].second << " Mat Size :" << scaledImages[scale].first.size() << " Dist:" << depth_avg << " Min/max:" << depth_min << "/" << depth_max;
#endif
        size_t thisWindowsChecked = 0;
        size_t thisWindowsPassed  = 0;

		vector<Window> unfilteredWindows;
        // Start at the upper left corner.  Loop through the rows and cols adding
		// each position to the list to check until the detection window falls off 
		// the edges of the scaled image
        for (int r = 0; (r + wsize) <= scaledImages[scale].first.rows; r += step)
        {
            for (int c = 0; (c + wsize) <= scaledImages[scale].first.cols; c += step)
            {
                thisWindowsChecked += 1;
				const Rect rect(c, r, wsize, wsize);
				unfilteredWindows.push_back(Window(rect, scale));
            }
        }
        debug_.initialWindows += thisWindowsChecked;
		vector<bool> validList;

		// If there is depth data, filter using it :
		// Throw out rects which would indicate an object that is at the
		// wrong depth given the size of the window being searched
		if (!depthIn.empty())
		{
			vector<MatT> depthList;

			for (size_t i = 0; i < unfilteredWindows.size(); i++)
			{
				const Rect r(unfilteredWindows[i].first);
				size_t scale = unfilteredWindows[i].second;
				depthList.push_back(scaledDepth[scale].first(r));
			}
			checkDepthList(depth_min, depth_max, depthList, validList);
		}
		else
		{
			validList = vector<bool>(unfilteredWindows.size(), true);
		}
		for (size_t i = 0; i < unfilteredWindows.size(); i++)
		{
			if (validList[i])
			{
				windows.push_back(unfilteredWindows[i]);
				thisWindowsPassed += 1;
			}
		}
#if 0
        cout << " Windows Passed:" << thisWindowsPassed << "/" << thisWindowsChecked << endl;
#endif
    }
	debug_.d12In = windows.size();
}


// Run the actual detection.  Pass in the classifer (a d12 or d24 
// neural net) along with a set of windows to search. The scaledImages
// vector is a set of the actual images to grab input data from.
// Threshold is the confidence limit we need to see to accept an
// image as detected, and label is the name of the object
// we're looking for.
// For each detected window, also return a score 
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::runDetection(ClassifierT &classifier,
                                  const vector<pair<MatT, double> >& scaledImages,
                                  const vector<Window>& windows,
                                  const float threshold,
                                  const string &label,
                                  vector<Window>& windowsOut,
                                  vector<float>& scores)
{
    windowsOut.clear();
    scores.clear();
    // Accumulate a number of images to test and pass them in to
    // the NN prediction as a batch
    vector<MatT> images;

    // Return value from detection. This is a list of indexes from
    // the input array above which have a high enough confidence score
    vector<size_t> detected;

    size_t batchSize = classifier.batchSize(); // defined when classifer is constructed
    int    counter   = 0;
    //double start     = gtod_wrapper(); // grab start time

    // For each input window, grab the correct image
    // subset from the correct scaled image.
    // Detection happens in batches, so save up a list of
    // images and submit them all at once.
    for (auto it = windows.cbegin(); it != windows.cend(); ++it)
    {
        // scaledImages[x].first is a Mat holding the image
        // scaled to the correct size for the given rect.
        // it->second is the index into scaledImages to look at
        // so scaledImages[it->second] is a Mat holding the original images 
		// resized to the correct scale for the current window. 
		// it->first is the rect describing the subset of that image 
		// we need to process
        images.push_back(scaledImages[it->second].first(it->first));
        if ((images.size() == batchSize) || ((it + 1) == windows.cend()))
        {
            doBatchPrediction(classifier, images, threshold, label, detected, scores);

            // Clear out images array to start the next batch
            // of processing fresh
            images.clear();

			// detected is a list of indexes of entries in the input
			// which returned confidences higher than threshold. Keep
			// those as valid detections and ignore the rest
            for (size_t j = 0; j < detected.size(); j++)
            {
                // Indexes in detected array are relative to the start of the
                // current batch just passed in to doBatchPrediction.
                // Use counter to keep track of which batch we're in
                windowsOut.push_back(windows[counter * batchSize + detected[j]]);
            }
            // Keep track of the batch number
            counter++;
        }
    }
    //double end = gtod_wrapper();
    //cout << "runDetection time = " << (end - start) << endl;
}


// do 1 run of the classifier. This takes up batch_size predictions
// and adds the index of anything found to the detected list
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::doBatchPrediction(ClassifierT &classifier,
												    const vector<MatT> &imgs,
												    const float         threshold,
												    const string&       label,
												    vector<size_t>&     detected,
												    vector<float>&      scores)
{
    detected.clear();
    // Grab the top 2 detected classes.  Since we're doing an object /
    // not object split, that will get the scores for both categories
    vector<vector<Prediction> > predictions = classifier.ClassifyBatch(imgs, 2);

    // Each outer loop is the predictions for one input image
    for (size_t i = 0; i < imgs.size(); ++i)
    {
        // Each inner loop is the prediction for a particular label
        // for the given image, sorted by score.
        //
        // Look for object with label <label>, >= <threshold> confidence
		// Higher confidences from the prediction mean that the net
		// thinks it is more likely that the label correctly
		// identifies the image passed in
        for (auto it = predictions[i].cbegin(); it != predictions[i].cend(); ++it)
        {
            if (it->first == label)
            {
                if (it->second >= threshold)
                {
                    detected.push_back(i);
                    scores.push_back(it->second);
                }
                break;
            }
        }
    }
}

// Use a specially-traned net to adjust the position of a detection
// rectangle to better match the actual position of the object we're
// looking for.
// The net is trained with a number of shifted and resized images. Each
// label 0-44 is some fixed permutation of shift and resize values.
// If the confidence for that label is high enough it means that
// the net thinks the ball is shifted/resized by the amount 
// corresponding to that label.  
// The actual shift/resize to apply is the average of all of them
// with a high enough confidence
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::runCalibration(const vector<Window>& windowsIn,
                                    const vector<pair<MatT, double> >& scaledImages,
                                    ClassifierT &classifier,
                                    const float threshold,
                                    vector<Window>& windowsOut)
{
	windowsOut.clear();
	vector<MatT>           images; // input images
	vector<vector<float> > shift;  // shift list for this batch
	vector<vector<float> > shifts; // complete list of all shifts
	for (auto it = windowsIn.cbegin(); it != windowsIn.cend(); ++it)
	{
		// Grab the rect from the scaled image represented
		// but each input window
		images.push_back(scaledImages[it->second].first(it->first));
		if ((images.size() == classifier.batchSize()) || ((it + 1) == windowsIn.cend()))
		{
			doBatchCalibration(classifier, images, threshold, shift);
			shifts.insert(shifts.end(), shift.begin(), shift.end());
			images.clear();
		}
	}
	for (size_t i = 0; i < windowsIn.size(); i++)
	{
		Rect rOut = windowsIn[i].first;
		// These are the inverse of the values used
		// to train the calibration net, since we want
		// to "unshift" the image back to the correct
		// location and size
		float ds = 1.0 / shifts[i][0];
		float dx = -shifts[i][1];
		float dy = -shifts[i][2];
#ifdef VERBOSE
		cout << "i = " << i << " In=" << rOut;
		cout << " ds=" << ds;
		cout << " dx=" << dx;
		cout << " dy=" << dy;
#endif
		// Actually apply the shift/scale
		rOut = Rect(rOut.tl().x - dx*rOut.width/ds, 
				    rOut.tl().y - dy*rOut.height/ds, 
					rOut.width/ds, 
					rOut.height/ds);
#ifdef VERBOSE
		cout << " Out=" << rOut;
#endif
		// Shift rectangles if they extend past the borders
		// of their respective scaledImages
		Size scaledImageSize = scaledImages[windowsIn[i].second].first.size();

		// Make sure new rect isn't resized to be larger
		// than the scaledImage size
		if (rOut.width > scaledImageSize.width)
		{
			rOut.x = 0;
			rOut.width = rOut.height = scaledImageSize.width;
#ifdef VERBOSE
			cout << " resized width to " << rOut << " to fit in " << scaledImageSize;
#endif
		}
		if (rOut.height > scaledImageSize.height)
		{
			rOut.y = 0;
			rOut.width = rOut.height = scaledImageSize.height;
#ifdef VERBOSE
			cout << " resized height to " << rOut << " to fit in " << scaledImageSize;
#endif
		}
		if(rOut.tl().x < 0)
		{
			rOut -= Point(rOut.tl().x, 0);
#ifdef VERBOSE
			cout << " Shifted X to 0:" << rOut << " size="<< scaledImageSize;
#endif
		}
		else if(rOut.br().x >= scaledImageSize.width)
		{
			rOut += Point(scaledImageSize.width - rOut.br().x, 0);
#ifdef VERBOSE
			cout << " Shifted X to max:" << rOut << " size="<< scaledImageSize;
#endif
		}
		if(rOut.tl().y < 0)
		{
			rOut -= Point(0, rOut.tl().y);
#ifdef VERBOSE
			cout << " Shifted Y to 0:" << rOut << " size="<< scaledImageSize;
#endif
		}
		else if(rOut.br().y >= scaledImageSize.height)
		{
			rOut += Point(0, scaledImageSize.height - rOut.br().y);
#ifdef VERBOSE
			cout << " Shifted Y to max:" << rOut << " br=" << rOut.br() << " size="<< scaledImageSize;
#endif
		}
		windowsOut.push_back(Window(rOut, windowsIn[i].second));
#ifdef VERBOSE
		cout << endl;
#endif
	}
}


template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::doBatchCalibration(ClassifierT            &classifier,
													 const vector<MatT>     &imgs,
													 const float             threshold,
													 vector<vector<float> > &shift)
{
	shift.clear();
	vector<vector<Prediction> > predictions = classifier.ClassifyBatch(imgs, 45);
	float ds[] = { .81, .93, 1, 1.10, 1.21 };
	float dx   = .17;
	float dy   = .17;
	// Each outer loop is the predictions for one input image
	for (size_t i = 0; i < imgs.size(); ++i)
	{
		// Each inner loop is the prediction for a particular label
		// for the given image, sorted by score.
		//
		// Look for object with label <label>, >= threshold confidence
		float dsc     = 0;
		float dxc     = 0;
		float dyc     = 0;
		int   counter = 0;
		for (auto it = predictions[i].cbegin(); it != predictions[i].cend(); ++it)
		{
			if (it->second >= threshold)
			{
				int index = stoi(it->first);
				dsc += ds[(index - index % 9) / 9]; //probably should just be index/9
				dxc += dx * (((index % 9) / 3) - 1);
				dyc += dy * (index % 3 - 1);
				counter++;
#ifdef VERBOSE
				cout << "i=" << i << " Label=" << it->first << " thresh=" << it->second ;
				cout << " ds idx=" << (index - index % 9) / 9;
				cout << " dx idx=" << (((index % 9) / 3) - 1);
				cout << " dy idx=" << (index % 3 - 1);
				cout << " ds=" << ds[(index-index%9)/9];
				cout << " dx=" << dx * (((index % 9) / 3) - 1);
				cout << " dy=" << dy * (index % 3 - 1);
				cout << " dsc=" << dsc;
				cout << " dx=" << dx;
				cout << " dy=" << dy;
				cout << endl;
#endif
			}
		}
		if(counter == 0)
		{
			dsc = 1;
			dxc = 0;
			dyc = 0;
		}
		else
		{
			dsc /= counter;
			dxc /= counter;
			dyc /= counter;
		}
		vector<float> shifts;
		shifts.push_back(dsc);
		shifts.push_back(dxc);
		shifts.push_back(dyc);
		shift.push_back(shifts);
	}
}


// Check each image in the list to see if any pixels
// in the window are at the correct depth for the size/scale
// of that window.  
template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::checkDepthList(const float depth_min, const float depth_max,
		const vector<Mat> &depthList, vector<bool> &validList)
{
	validList.clear();
	for (auto it = depthList.cbegin(); it != depthList.cend(); ++it)
		validList.push_back(depthInRange(depth_min, depth_max, *it));
}


// Be conservative here - if any of the depth values in the target rect
// are in the expected range, consider the rect in range.  Also
// say that it is in range if any of the depth values are negative (i.e. no
// depth info for those pixels)
template<class MatT, class ClassifierT>
bool NNDetect<MatT, ClassifierT>::depthInRange(const float depth_min, const float depth_max, const Mat& detectCheck)
{
    for (int py = 0; py < detectCheck.rows; py++)
    {
        const float *p = detectCheck.ptr<float>(py);
        for (int px = 0; px < detectCheck.cols; px++)
        {
            if (std::isnan(p[px]) || (p[px] <= 0.0) || ((p[px] < depth_max) && (p[px] > depth_min)))
            {
                return true;
            }
        }
    }
    return false;
}

// GPU specialization
vector<bool> cudaDepthThreshold(const vector<GpuMat> &depthList, const float depthMin, const float depthMax);

template<class MatT, class ClassifierT>
void NNDetect<MatT, ClassifierT>::checkDepthList(const float depth_min, const float depth_max,
		const vector<GpuMat> &depthList, vector<bool> &validList)
{
	const size_t batchSize = 128;
	validList.clear();
	vector<GpuMat> depthBatch;
	vector<Mat> depthMats;
	for (auto it = depthList.cbegin(); it != depthList.cend(); ++it)
	{
		depthBatch.push_back(*it);
		if ((depthBatch.size() == batchSize) || (it == (depthList.cend() - 1)))
		{
			auto validBatch = cudaDepthThreshold(depthBatch, depth_min, depth_max);
			for (auto v = validBatch.cbegin(); v != validBatch.cend(); ++v)
				validList.push_back(*v);

			depthBatch.clear();
			validBatch.clear();
		}
	}
}

template<class MatT, class ClassifierT>
bool NNDetect<MatT, ClassifierT>::initialized(void) const
{
	return d12_.initialized() && d24_.initialized() &&
	        c12_.initialized() && c24_.initialized();
}

template<class MatT, class ClassifierT>
NNDetectDebugInfo NNDetect<MatT, ClassifierT>::DebugInfo(void) const
{
	return debug_;
}

// Explicitly instatiate classes used elsewhere
#ifndef USE_GIE
template class NNDetect<Mat, CaffeClassifier<Mat>>;
template class NNDetect<GpuMat, CaffeClassifier<GpuMat>>;
#else
template class NNDetect<Mat, GIEClassifier<Mat>>;
template class NNDetect<GpuMat, GIEClassifier<GpuMat>>;
#endif
