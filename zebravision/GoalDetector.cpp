#include <iomanip>
#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

#define VERBOSE

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui) :
	_fov_size(fov_size),
	_frame_size(frame_size),
	_isValid(false),
	_pastRects(2),
	_min_valid_confidence(0.25),
	_otsu_threshold(5),
	_blue_scale(70),
	_red_scale(50),
	_camera_angle(530)
{
	if (gui)
	{
		cv::namedWindow("Goal Detect Adjustments", CV_WINDOW_NORMAL);
		createTrackbar("Blue Scale","Goal Detect Adjustments", &_blue_scale, 100);
		createTrackbar("Red Scale","Goal Detect Adjustments", &_red_scale, 100);
		createTrackbar("Otsu Threshold","Goal Detect Adjustments", &_otsu_threshold, 255);
		createTrackbar("Camera Angle","Goal Detect Adjustments", &_camera_angle, 900);
	}
}

bool GoalDetector::Valid(void) const
{
	return _isValid;
}
// Compute a confidence score for an actual measurement given
// the expected value and stddev of that measurement
// Values around 0.5 are good. Values away from that are progressively
// worse.  Wrap stuff above 0.5 around 0.5 so the range
// of values go from 0 (bad) to 0.5 (good).
float GoalDetector::createConfidence(float expectedVal, float expectedStddev, float actualVal)
{
	pair<float,float> expectedNormal(expectedVal, expectedStddev);
	float confidence = utils::normalCFD(expectedNormal, actualVal);
	return confidence > 0.5 ? 1 - confidence : confidence;
}


void GoalDetector::findBoilers(const cv::Mat& image, const cv::Mat& depth) {
	//ObjectType(4) == top piece of tape
	//ObjectType(5) == bottom piece of tape
	clear();
	vector<vector<Point>> goal_contours = getContours(image);
	vector<float> top_goal_depths = getDepths(depth,goal_contours,4, ObjectType(4).real_height());
	vector<float> bottom_goal_depths = getDepths(depth,goal_contours,5, ObjectType(5).real_height());
	//compute confidences for both the top piece of tape and the bottom piece of tape
	vector<GoalInfo> top_info = getInfo(goal_contours,top_goal_depths,4);
	if(top_info.size() == 0)
		return;
	vector<GoalInfo> bottom_info = getInfo(goal_contours,bottom_goal_depths,5);
	if(bottom_info.size() == 0)
		return;
	cout << top_info.size() << " top goals found and " << bottom_info.size() << " bottom" << endl;	

	int best_result_index_top = 0;
	int best_result_index_bottom = 0;
	bool found_goal = false;
	bool initialized = false;
	//loop through every combination of top and bottom goal and check for the following conditions:
	//top is above bottom
	//confidences are higher than any previous one
	for(size_t i = 0; i < top_info.size(); i++) {
		for(size_t j = 0; j < bottom_info.size(); j++) {
			if(top_info[i].pos.z < bottom_info[j].pos.z || top_info[i].vec_index == bottom_info[j].vec_index || abs(top_info[i].angle - bottom_info[j].angle) > 10.0)
				continue;
			if(!initialized) {
				best_result_index_top = i;
				best_result_index_bottom = j;
				initialized = true;
				}
			if(top_info[best_result_index_top].confidence + bottom_info[best_result_index_bottom].confidence <= top_info[i].confidence + bottom_info[j].confidence) {
				found_goal = true;
				best_result_index_top = i;
				best_result_index_bottom = j;
			}
		}
	}
	cout << "Top distance: " << top_info[best_result_index_top].distance << " Bottom distance: " << bottom_info[best_result_index_bottom].distance << endl;
	cout << "Top position: " << top_info[best_result_index_top].pos << " Bottom position: " << bottom_info[best_result_index_bottom].pos << endl;
	cout << "Top confidence: " << top_info[best_result_index_top].confidence << " Bottom confidence: " << bottom_info[best_result_index_bottom].confidence << endl;
	cout << "Found Goal: " << found_goal << endl;
	
	//say a goal is found if the sum of the confidences is higher than 0.5
	if(found_goal && top_info[best_result_index_top].confidence + bottom_info[best_result_index_bottom].confidence > 0.3) {
		cout << "Found goal with confidence: " << top_info[best_result_index_top].confidence + bottom_info[best_result_index_bottom].confidence << endl;
		_pastRects.push_back(SmartRect(top_info[best_result_index_top].rect));
		_goal_pos      = top_info[best_result_index_top].pos;
		_dist_to_goal  = top_info[best_result_index_top].distance; 
		_angle_to_goal = top_info[best_result_index_top].angle;
		_goal_top_rect     = top_info[best_result_index_top].rect;
		_goal_bottom_rect = bottom_info[best_result_index_bottom].rect;
	} else
		_pastRects.push_back(SmartRect(Rect()));
	
	isValid();
}


void GoalDetector::clear()
{

	// Reset previous detection vars
	_isValid = false;
	_dist_to_goal = -1.0;
	_angle_to_goal = -1.0;
	_goal_top_rect = Rect();
	_goal_bottom_rect = Rect();
	_goal_pos  = Point3f();


}

vector< vector < Point > > GoalDetector::getContours(const Mat& image) {
	// Look for parts the the image which are within the
	// expected bright green color range
	Mat threshold_image;
	vector < vector < Point > > return_contours;
	if (!generateThresholdAddSubtract(image, threshold_image))
	{
		_pastRects.push_back(SmartRect(Rect()));
		return return_contours;
	}

	// find contours in the thresholded image - these will be blobs
	// of green to check later on to see how well they match the
	// expected shape of the goal
	// Note : findContours modifies the input mat
	Mat threshold_copy = threshold_image.clone();
	vector<Vec4i>          hierarchy;
	findContours(threshold_copy, return_contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	return return_contours;
}

vector< float > GoalDetector::getDepths(const Mat &depth, vector< vector< Point > > contours, int objtype, float expected_height) {
	// Use to mask the contour off from the rest of the
	// image - used when grabbing depth data for the contour
	Mat contour_mask(_frame_size, CV_8UC1, Scalar(0));
	vector< float > return_vec;
	for(size_t i = 0; i < contours.size(); i++) {
		// get the minimum and maximum depth values in the contour,
		Rect br(boundingRect(contours[i]));
		Moments mu = moments(contours[i], false);
		Point com = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
		//Point center(rect.tl().x+rect.size().width/2, rect.tl().y+rect.size().height/2);
		
		//create a mask which is the same shape as the contour
		contour_mask.setTo(Scalar(0));
		drawContours(contour_mask, contours, i, Scalar(255), CV_FILLED);
		// copy them into individual floats
		pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, br, 10);
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;
		cout << "Depth " << depth_z_min << " " << depth_z_max << endl;

		// If no depth data, calculate it using FOV and height of
		// the target. This isn't perfect but better than nothing
		if ((depth_z_min <= 0.) || (depth_z_max <= 0.)) {
			ObjectType ot(objtype);
			depth_z_min = depth_z_max = distanceUsingFixedHeight(br,com,expected_height);
		}
		return_vec.push_back(depth_z_max);
	}
	return return_vec;
}

vector<GoalInfo> GoalDetector::getInfo(vector<vector<Point>> _contours,vector<float> _depth_maxs, int objtype) {
	ObjectType _goal_shape(objtype);
	vector<GoalInfo> return_info;
	// Create some target stats based on our idealized goal model
	//center of mass as a percentage of the object size from top left
	const Point2f com_percent_expected(_goal_shape.com().x / _goal_shape.width(),
	_goal_shape.com().y / _goal_shape.height());
	// Ratio of contour area to bounding box area
	const float filledPercentageExpected = _goal_shape.area() / _goal_shape.boundingArea();

	// Aspect ratio of the goal
	const float expectedRatio = _goal_shape.width() / _goal_shape.height();

	vector<GoalInfo> best_goals;
	for (size_t i = 0; i < _contours.size(); i++)
	{
		// ObjectType computes a ton of useful properties so create
		// one for what we're looking at
		Rect br(boundingRect(_contours[i]));

		// Remove objects which are obviously too small
		// TODO :: Tune me, make me a percentage of screen area?
		if ((br.area() <= 250.0))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " area out of range " << br.area() << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		// Remove objects too low on the screen
		/*if (br.br().y > (_frame_size.width * 0.4f))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " br().y out of range "<< br.br().y << endl;
#endif
			_confidence.push_back(0);
			continue;
		}*/


		
		// TODO : Figure out how well this works in practice
		// Filter out goals which are too close or too far
		if ( 6.2 <_depth_maxs[i] || _depth_maxs[i] < .1)
		{
#ifdef VERBOSE
			cout << "Contour " << i << " depth out of range " << _depth_maxs[i] << endl;
#endif
			_confidence.push_back(0);
			continue;
		}


		//create a trackedobject to get various statistics
		//including area and x,y,z position of the goal
		ObjectType goal_actual(_contours[i], "Actual Goal", 0);
		TrackedObject goal_tracked_obj(0, _goal_shape, br, _depth_maxs[i], _fov_size, _frame_size,-((float)_camera_angle/10.) * M_PI / 180.0);
		//TrackedObject goal_tracked_obj(0, _goal_shape, br, _depth_maxs[i], _fov_size, _frame_size, -16 * M_PI / 180.0);

		// Gets the bounding box area observed divided by the
		// bounding box area calculated given goal size and distance
		// For an object the size of a goal we'd expect this to be
		// close to 1.0 with some variance due to perspective
		float exp_area = goal_tracked_obj.getScreenPosition(_fov_size, _frame_size).area();
		float actualScreenArea = (float)br.area() / exp_area;
/*
		if (((exp_area / br.area()) < 0.20) || ((exp_area / br.area()) > 5.00))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " area out of range for depth (depth/act/exp/ratio):" << _depth_maxs[i] << "/" << br.area() << "/" << exp_area << "/" << actualScreenArea << endl;
#endif
			_confidence.push_back(0);
			continue;
		}*/
		//percentage of the object filled in
		float filledPercentageActual = goal_actual.area() / goal_actual.boundingArea();

		//center of mass as a percentage of the object size from top left
		Point2f com_percent_actual((goal_actual.com().x - br.tl().x) / goal_actual.width(),
								   (goal_actual.com().y - br.tl().y) / goal_actual.height());

		//width to height ratio
		float actualRatio = goal_actual.width() / goal_actual.height();

		/* I don't think this block of code works but I'll leave it in here
		Mat test_contour = Mat::zeros(640,640,CV_8UC1);
		std::vector<Point> upscaled_contour;
		for(int j = 0; j < _goal_shape.shape().size(); j++) {
			upscaled_contour.push_back(Point(_goal_shape.shape()[j].x * 100, _goal_shape.shape()[j].y * 100));
			cout << "Upscaled contour point: " << Point(_goal_shape.shape()[j].x * 100, _goal_shape.shape()[j].y * 100) << endl;
			} 
		std::vector< std::vector<Point> > upscaled_contours;
		upscaled_contours.push_back(upscaled_contour);
		drawContours(test_contour, upscaled_contours, 0, Scalar(0,0,0));
		imshow("Goal shape", test_contour);
		*/

		//parameters for the normal distributions
		//values for standard deviation were determined by
		//taking the standard deviation of a bunch of values from the goal
		//confidence is near 0.5 when value is near the mean
		//confidence is small or large when value is not near mean
		float confidence_height      = createConfidence(_goal_shape.real_height(), 0.4, goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0);
		float confidence_com_x       = createConfidence(com_percent_expected.x, 0.125,  com_percent_actual.x);
		float confidence_filled_area = createConfidence(filledPercentageExpected, 0.33,   filledPercentageActual);
		float confidence_ratio       = createConfidence(expectedRatio, 4,  actualRatio);
		float confidence_screen_area = createConfidence(1.0, 0.75,  actualScreenArea);

		// higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_filled_area + confidence_ratio/2. + confidence_screen_area/2.) / 5.0;
		_confidence.push_back(confidence);

#ifdef VERBOSE
		cout << "-------------------------------------------" << endl;
		cout << "Contour " << i << endl;
		cout << "confidence_height: " << confidence_height << endl;
		cout << "confidence_com_x: " << confidence_com_x << endl;
		cout << "confidence_filled_area: " << confidence_filled_area << endl;
		cout << "confidence_ratio: " << confidence_ratio << endl;
		cout << "confidence_screen_area: " << confidence_screen_area << endl;
		cout << "confidence: " << confidence << endl;
		cout << "Height exp/act: " << _goal_shape.real_height() << "/" <<  goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0 << endl;
		cout << "Depth max: " << _depth_maxs[i] << endl;
		cout << "Area exp/act: " << (int)exp_area << "/" << br.area() << endl;
		cout << "Aspect ratio exp/act : " << expectedRatio << "/" << actualRatio << endl;
		cout << "br.br().y: " << br.br().y << endl;
		cout << "position: " << goal_tracked_obj.getPosition() << endl;
		cout << "-------------------------------------------" << endl;
#endif

		// This goal passes the threshold required for us to consider it a goal
		// Add it to the list of best goals
		GoalInfo goal_info;

		goal_info.pos        = goal_tracked_obj.getPosition();
		goal_info.confidence = confidence;
		goal_info.distance   = _depth_maxs[i] * cosf((_camera_angle/10.0) * (M_PI/180.0));
		goal_info.angle 	 = atan2f(goal_info.pos.x, goal_info.pos.y) * 180. / M_PI;
		goal_info.rect   	 = br;
		goal_info.vec_index = i;
		return_info.push_back(goal_info);
	}
	return return_info;
}


// We're looking for pixels which are mostly green
// with a little bit of blue - that should match
// the LED reflected color.
// Do this by splitting channels and combining
// them into one grayscale channel.
// Start with the green value.  Subtract the red
// channel - this will penalize pixels which have red
// in them, which is good since anything with red
// is an area we should be ignoring. Do the same with
// blue, except multiply the pixel values by a weight
// < 1. Using this weight will let blue-green pixels
// show up in the output grayscale
bool GoalDetector::generateThresholdAddSubtract(const Mat& imageIn, Mat& imageOut)
{
    vector<Mat> splitImage;
    Mat         bluePlusRed;

    split(imageIn, splitImage);
	addWeighted(splitImage[0], _blue_scale / 100.0,
			    splitImage[2], _red_scale / 100.0, 0.0,
				bluePlusRed);
	subtract(splitImage[1], bluePlusRed, imageOut);

    Mat erodeElement(getStructuringElement(MORPH_RECT, Size(3, 3)));
    Mat dilateElement(getStructuringElement(MORPH_RECT, Size(3, 3)));
	for (int i = 0; i < 2; ++i)
	{
		erode(imageOut, imageOut, erodeElement, Point(-1, -1), 1);
		dilate(imageOut, imageOut, dilateElement, Point(-1, -1), 1);
	}

	// Use Ostu adaptive thresholding.  This will turn
	// the gray scale image into a binary black and white one, with pixels
	// above some value being forced white and those below forced to black
	// The value to used as the split between black and white is returned
	// from the function.  If this value is too low, it means the image is
	// really dark and the returned threshold image will be mostly noise.
	// In that case, skip processing it entirely.
	double otsuThreshold = threshold(imageOut, imageOut, 0., 255., CV_THRESH_BINARY | CV_THRESH_OTSU);
#ifdef VERBOSE
	cout << "OSTU THRESHOLD " << otsuThreshold << endl;
#endif
	if (otsuThreshold < _otsu_threshold)
		return false;
    return countNonZero(imageOut) != 0;
}

// Use the camera FOV, image size and rect size to
// estimate distance to a target
float GoalDetector::distanceUsingFOV(ObjectType _goal_shape, const Rect &rect) const
{
	float percent_image = (float)rect.height / _frame_size.height;
	float size_fov = percent_image * _fov_size.y; //TODO fov size
	return _goal_shape.height() / (2.0 * tanf(size_fov / 2.0));
}

float GoalDetector::distanceUsingFixedHeight(const Rect &rect, Point center , float expected_delta_height) const {
	/*
	cout << "Center: " << center << endl;
	float percent_image = ((float)center.y - (float)_frame_size.height/2.0)  / (float)_frame_size.height;
	cout << "Percent Image: " << percent_image << endl;
	cout << "FOV Size: " << _fov_size << endl;
	float size_fov = (percent_image * 1.3714590199999497) + (((float)_camera_angle/10.0) * (M_PI/180.0));
	cout << "Size FOV: " << size_fov << endl;
	cout << "Depth: " << expected_delta_height / (2.0 * tanf(size_fov/2.0)) << endl;
	return expected_delta_height / (2.0 * tanf(size_fov / 2.0));*/

	/*float focal_length_px = 0.5 * _frame_size.height / tanf(_fov_size.y / 2.0);
	cout << "Focal Length: " << focal_length_px << endl;
	float distance_centerline = (expected_delta_height * focal_length_px) / ((float)center.y - (float)_frame_size.height/2);
	cout << "Distance to centerline: " << distance_centerline << endl;
	float distance_ground = cos((_camera_angle/10.0) * (M_PI/180.0)) * distance_centerline;
	cout << "Distance to ground: " << distance_ground << endl;
	return distance_ground; */

	//float focal_length_px = 0.5 * _frame_size.height / tanf(0.777 / 2.0);
	float focal_length_px = 750.0;
	float to_center = (402.2 - (float)center.y);
	float distance_diagonal = (focal_length_px * expected_delta_height) / (focal_length_px * sin((_camera_angle/10.0) * (M_PI/180.0)) + to_center);
	return distance_diagonal;
}

float GoalDetector::dist_to_goal(void) const
{
 	//floor distance to goal in m
	return _isValid ? _dist_to_goal * 1.1 : -1.0;
}

float GoalDetector::angle_to_goal(void) const
{
	//angle robot has to turn to face goal in degrees
	if (!_isValid)
		return -1;

	float delta = 0;
	if (_angle_to_goal >= 40)
		delta = -3.00; // >= 40
	else if (_angle_to_goal >= 35)
		delta = -2.50; // 35 < x <= 40
	else if (_angle_to_goal >= 30)
		delta = -2.00; // 30 < x <= 35
	else if (_angle_to_goal >= 25)
		delta = -0.50; // 25 < x <= 30
	else if (_angle_to_goal >= 20)
		delta = -0.15; // 20 < x <= 25
	else if (_angle_to_goal >= -20)
		delta = 0;     // -20 <= x <= 20
	else if (_angle_to_goal >= -25)
		delta = 0.25;  // -25 <= x < -20
	else if (_angle_to_goal >= -30)
		delta = 0.50;  // -30 <= x < -25
	else if (_angle_to_goal >= -35)
		delta = 2.50;  // -35 <= x < -30
	else if (_angle_to_goal >= -40)
		delta = 3.50;  // -40 <= x < -35
	else
		delta = 4.55;  // -40 > x 

	cout << "angle:" << _angle_to_goal << " delta:" << delta << endl;

	return _angle_to_goal + delta;
}

// Screen rect bounding the goal
Rect GoalDetector::goal_rect(void) const
{
	return _isValid ? _goal_top_rect : Rect();
}

// Goal x,y,z position relative to robot
Point3f GoalDetector::goal_pos(void) const
{
	return _isValid ? _goal_pos : Point3f();
}

// Draw debugging info on frame - all non-filtered contours
// plus their confidence. Highlight the best bounding rect in
// a different color
void GoalDetector::drawOnFrame(Mat &image, vector<vector<Point>> _contours) const
{
	for (size_t i = 0; i < _contours.size(); i++)
	{
		drawContours(image, _contours, i, Scalar(0,0,255), 3);
		Rect br(boundingRect(_contours[i]));
		rectangle(image, br, Scalar(255,0,0), 2);
//		stringstream confStr;
//		confStr << fixed << setprecision(2) << _confidence[i];
//		putText(image, confStr.str(), br.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));
//		putText(image, to_string(i), br.br(), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
	}
	if(!(_pastRects[_pastRects.size() - 1] == SmartRect(Rect()))) {
		rectangle(image, _goal_top_rect, Scalar(0,255,0), 2);	
		rectangle(image, _goal_bottom_rect, Scalar(0,140,255), 2);	
	}
}

// Look for the N most recent detected rectangles to be
// the same before returning them as valid. This makes sure
// the camera has stopped moving and has settled
// TODO : See if we want to return different values for
// several frames which have detected goals but at different
// locations vs. several frames which have no detection at all
// in them?
void GoalDetector::isValid()
{
	SmartRect currentRect = _pastRects[0];
#if 0
	for(auto it = _pastRects.begin() + 1; it != _pastRects.end(); ++it)
	{
		if(!(*it == currentRect))
		{
			_isValid = false;
			return;
		}
	}
#endif
	if (currentRect == SmartRect(Rect()))
		_isValid = false;
	else
		_isValid = true;
}

// Simple class to encapsulate a rect plus a slightly
// more complex than normal comparison function
SmartRect::SmartRect(const cv::Rect &rectangle):
   myRect(rectangle)
{
}

// Overload equals. Make it so that empty rects never
// compare true, even to each other.
// Also, allow some minor differences in rectangle positions
// to still count as equivalent.
bool SmartRect::operator== (const SmartRect &thatRect)const
{
	if(myRect == Rect() || thatRect.myRect == Rect())
	{
		return false;
	}
	double intersectArea = (myRect & thatRect.myRect).area();
	double unionArea     = myRect.area() + thatRect.myRect.area() - intersectArea;

	return (intersectArea / unionArea) >= .8;
}

std::ostream& operator<<(std::ostream& os, const SmartRect &obj) {
	if(obj.myRect == Rect())
		os << "NULL";
	else
		os << obj.myRect;
	return os;
}
