#pragma once
//standard include
#include <math.h>
#include <iostream>

//opencv include
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <boost/circular_buffer.hpp>

#include "Utilities.hpp"
#include "track3d.hpp"

class SmartRect
{
    public:
        SmartRect(const cv::Rect &myRect);
        cv::Rect myRect;
        bool operator== (const SmartRect &thatRect)const;
};

//this contains all the info we need to decide between goals once we are certain if it is a goal
struct GoalInfo
{
	bool operator () (GoalInfo a, GoalInfo b) {
		return (a.confidence > b.confidence);
	}
	cv::Point3f pos;
	float confidence;
	float distance;
	float angle;
	cv::Rect rect;
};

class GoalDetector
{
	public:
		GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui = false);

		float dist_to_goal(void) const;
		float angle_to_goal(void) const;
		cv::Rect goal_rect(void) const;
		cv::Point3f goal_pos(void) const;
		void drawOnFrame(cv::Mat &image) const;

		//These are the three functions to call to run GoalDetector
		//they fill in _contours, _infos, _confidence, _depth_mins, etc
		void clear(void);
		//If your objectypes have the same width it's safe to run
		//getContours and computeConfidences with different types
		void findBoilers(const cv::Mat& image, const cv::Mat& depth);
		void getContours(int objtype, const cv::Mat& image, const cv::Mat& depth);
		void computeConfidences(int objtype);		
	private:
	
		cv::Point2f _fov_size;
		cv::Size _frame_size;
		const float _goal_height = 1.524;  // TODO : remeasure me!

		//const float _goal_height = .5f;

		// Save detection info
		bool _isValid;
		boost::circular_buffer<SmartRect> _pastRects;
		float _dist_to_goal;
		float _angle_to_goal;
		cv::Rect _goal_rect;
		cv::Point3f _goal_pos;

		// Save all contours found in case we want to display
		std::vector<std::vector<cv::Point> > _contours;
		std::vector<float> _confidence;
		std::vector<GoalInfo> _infos;
		std::vector< float > _depth_maxs;
		std::vector< float > _depth_mins;

		float _min_valid_confidence;

		int   _otsu_threshold;
		int   _blue_scale;
		int   _red_scale;

int _camera_angle;

		float createConfidence(float expectedVal, float expectedStddev, float actualVal);
		float distanceUsingFOV(ObjectType _goal_shape, const cv::Rect &rect) const;
		bool generateThresholdAddSubtract(const cv::Mat& imageIn, cv::Mat& imageOut);
		void isValid();
};
