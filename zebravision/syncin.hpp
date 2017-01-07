// Input class to handle RGB (or grayscale?) video inputs :
// MPG, AVI, MP4, etc.
// Update needs to stay in sync with getFrame() calls to insure
// all frames are processed by main thread
// Code runs a separate decode thread which tries to buffer
// one frame ahead of the data needed by getFrame
#pragma once

#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mediain.hpp"

class ZvSettings;

class SyncIn : public MediaIn
{
	public:
		SyncIn(ZvSettings *settings = NULL);

		bool getFrame(cv::Mat &frame, cv::Mat &depth, bool pause = false);
		void frameNumber(int framenumber);

	protected:
		// Derived classes need to start and stop
		// the update thread. startThread should be
		// called once the input source is ready to 
		// send data and stop should be called in the
		// destructor
		void startThread(void);
		void stopThread(void);

		// Defined in derived classes to handle the nuts
		// and bolts of grabbing a frame from a given
		// source.  preLock happens before the mutex
		// while postLock happens inside it
		virtual bool postLockUpdate(cv::Mat &frame, cv::Mat &depth) = 0;
		virtual bool postLockFrameNumber(int framenumber) = 0;

	private:
		// frame_ is the most recent frame grabbed from 
		// the camera
		// prevGetFrame_ is the last frame returned from
		// getFrame().  If paused, code needs to keep returning
		// this frame rather than getting a new one from frame_
		cv::Mat           frame_;
		cv::Mat           depth_;
		cv::Mat           prevGetFrame_;
		cv::Mat           prevGetDepth_;

		// Mutex used to protect frame_
		// from simultaneous accesses 
		// by multiple threads
		boost::mutex      mtx_;
		
		// Condition variable used to signal between
		// update & getFrame - communicates when a 
		// frame is ready to use or needs to be read
		boost::condition_variable condVar_;
		//
		// Flag used to syncronize between update and get calls
		bool             frameReady_;

		// Thread object to track update() thread
		boost::thread    thread_;

		void update(void);
};
