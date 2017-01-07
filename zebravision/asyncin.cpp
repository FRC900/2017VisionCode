#include <iostream>

#include "asyncin.hpp"

using namespace cv;
using namespace std;

// Nothing interesting here, just pass settings through
// to the base class
AsyncIn::AsyncIn(ZvSettings *settings) :
	MediaIn(settings),
	updateStarted_(false)
{
}


// Start update thread. Would be nice to put in
// constructor but need to wait until after input
// source is initialized
void AsyncIn::startThread(void)
{
	thread_ = boost::thread(&AsyncIn::update, this);
}

void AsyncIn::stopThread(void)
{
	thread_.interrupt();
	thread_.join();
}

// update thread.  Run non-stop grabbing frames
// from input source and storing them in
// frame_ and depth_.  Don't worry about 
void AsyncIn::update(void)
{
	bool good = true;
	while (good)
	{
		boost::this_thread::interruption_point();
		FPSmark();

		if (!isOpened() || !preLockUpdate())
			good = false;

		boost::lock_guard<boost::mutex> guard(mtx_);

		// Now have exclusive access to frame_
		// and depth_.  Update them from the input
		// source here
		if (!good || !postLockUpdate(frame_, depth_))
		{
			good = false;
			frame_ = Mat();
			depth_ = Mat();
		}
		else
		{
			setTimeStamp();
			incFrameNumber();
			while (frame_.rows > 700)
			{
				pyrDown(frame_, frame_);
				if (!depth_.empty())
					pyrDown(depth_, depth_);
			}
		}

		// Signal that update loop has made it through
		// at least once. Prevents 1st getFrame() call
		// from returning empty because 1st update()
		// didn't finish first
		updateStarted_ = true;
		condVar_.notify_all();
	}
}


bool AsyncIn::getFrame(Mat &frame, Mat &depth, bool pause)
{
	if (!isOpened())
		return false;
	
	if (!pause)
	{
		// Make sure only one thread is accessing
		// shared frame_ and depth_ buffers at once
		boost::mutex::scoped_lock guard(mtx_);

		// Only needed to make sure the first
		// frame is processed in update() before
		// grabbing it here
		while (!updateStarted_)
			condVar_.wait(guard);

		// Use an empty mat to signal an error 
		// happened in update
		if (frame_.empty())
			return false;

		// Lock in the time and frame number associated
		// with depth_ and frame_ so they're returned when
		// queried from the main code
		lockTimeStamp();
		lockFrameNumber();
		frame_.copyTo(pausedFrame_);
		depth_.copyTo(pausedDepth_);
	}

	// Use an empty mat to signal an error 
	// happened in update
	if (pausedFrame_.empty())
		return false;

	pausedFrame_.copyTo(frame);
	pausedDepth_.copyTo(depth);
	return true;
}

