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

#if 0
class SmartRect
{
    public:
        SmartRect(const cv::Rect &myRect);
        cv::Rect myRect;
        bool operator== (const SmartRect &thatRect)const;
	friend std::ostream& operator<<(std::ostream& os, const SmartRect& obj);	
};
#endif

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
	size_t vec_index;
};

class GoalDetector
{
	public:
		GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui = false);

		float dist_to_goal(void) const;
		float angle_to_goal(void) const;
		cv::Rect goal_rect(void) const;
		cv::Point3f goal_pos(void) const;
		void drawOnFrame(cv::Mat &image,std::vector< std::vector< cv::Point>> _contours) const;

		//These are the three functions to call to run GoalDetector
		//they fill in _contours, _infos, _depth_mins, etc
		void clear(void);
		//If your objectypes have the same width it's safe to run
		//getContours and computeConfidences with different types
		void findBoilers(const cv::Mat& image, const cv::Mat& depth);
		std::vector< std::vector< cv::Point > > getContours(const cv::Mat& image);
		std::vector< float > getDepths(const cv::Mat &depth, std::vector< std::vector< cv::Point > > contours, int objtype, float expected_height);
		std::vector< GoalInfo > getInfo(std::vector< std::vector< cv::Point > > _contours, std::vector< float > _depth_maxs, int objtype);

		bool Valid(void) const;
	private:
	
		cv::Point2f _fov_size;
		cv::Size _frame_size;

		// Save detection info
		bool _isValid;
		//boost::circular_buffer<SmartRect> _pastRects;
		float _dist_to_goal;
		float _angle_to_goal;
		cv::Rect _goal_top_rect;
		cv::Rect _goal_bottom_rect;
		cv::Point3f _goal_pos;

		float _min_valid_confidence;

		int   _otsu_threshold;
		int   _blue_scale;
		int   _red_scale;

		int _camera_angle;

		float createConfidence(float expectedVal, float expectedStddev, float actualVal);
		float distanceUsingFOV(ObjectType _goal_shape, const cv::Rect &rect) const;
		float distanceUsingFixedHeight(const cv::Rect &rect,const cv::Point &center, float expected_delta_height) const;
		bool generateThresholdAddSubtract(const cv::Mat& imageIn, cv::Mat& imageOut);
		void isValid();
};
