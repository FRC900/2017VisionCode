#include <iomanip>
#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

#define VERBOSE

GoalDetector::GoalDetector(int obj_type, cv::Point2f fov_size, cv::Size frame_size, bool gui) :
	_goal_shape(obj_type),
	_fov_size(fov_size),
	_frame_size(frame_size),
	_isValid(false),
	_pastRects(2),
	_min_valid_confidence(0.25),
	_otsu_threshold(5),
	_blue_scale(87),
	_red_scale(60),
	_camera_angle(90)
{
	if (gui)
	{
		cv::namedWindow("Goal Detect Adjustments", CV_WINDOW_NORMAL);
		createTrackbar("Blue Scale","Goal Detect Adjustments", &_blue_scale, 100);
		createTrackbar("Red Scale","Goal Detect Adjustments", &_red_scale, 100);
		createTrackbar("Otsu Threshold","Goal Detect Adjustments", &_otsu_threshold, 255);
		createTrackbar("Camera Angle","Goal Detect Adjustments", &_camera_angle, 255);
	}
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

//this contains all the info we need to decide between goals once we are certain if it is a goal
struct GoalInfo
{
	Point3f pos;
	float confidence;
	float distance;
	float angle;
	Rect rect;
};


void GoalDetector::processFrame(const Mat& image, const Mat& depth)
{
	vector<GoalInfo> best_goals_updated;
	// Use to mask the contour off from the rest of the
	// image - used when grabbing depth data for the contour
	Mat contour_mask(image.rows, image.cols, CV_8UC1, Scalar(0));

	// Reset previous detection vars
	_isValid = false;
	_dist_to_goal = -1.0;
	_angle_to_goal = -1.0;
	_goal_rect = Rect();
	_goal_pos  = Point3f();
	_confidence.clear();
	_contours.clear();

	// Look for parts the the image which are within the
	// expected bright green color range
	Mat threshold_image;
	if (!generateThresholdAddSubtract(image, threshold_image))
	{
		_pastRects.push_back(SmartRect(Rect()));
		return;
	}

	// find contours in the thresholded image - these will be blobs
	// of green to check later on to see how well they match the
	// expected shape of the goal
	// Note : findContours modifies the input mat
	Mat threshold_copy = threshold_image.clone();
	vector<Vec4i>          hierarchy;
	findContours(threshold_copy, _contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

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
		if ((br.area() <= 350.0) || (br.area() > 8500))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " area out of range " << br.area() << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		// Remove objects too low on the screen
		if (br.br().y > (image.rows * 0.7f))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " br().y out of range "<< br.br().y << endl;
#endif
			_confidence.push_back(0);
			continue;
		}


		//create a mask which is the same shape as the contour
		contour_mask.setTo(Scalar(0));
		drawContours(contour_mask, _contours, i, Scalar(255), CV_FILLED);
	
		// get the minimum and maximum depth values in the contour,
		// copy them into individual floats
		pair<float, float> minMax = utils::minOfDepthMat(depth, contour_mask, br, 10);
		float depth_z_min = minMax.first;
		float depth_z_max = minMax.second;

		// If no depth data, calculate it using FOV and height of
		// the target. This isn't perfect but better than nothing
		if ((depth_z_min <= 0.) || (depth_z_max <= 0.))
			depth_z_min = depth_z_max = distanceUsingFOV(br);
		
		// TODO : Figure out how well this works in practice
		// Filter out goals which are too close or too far
		if ((depth_z_max < 1.) || (depth_z_min > 6.2))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " depth out of range "<< depth_z_min << " / " << depth_z_max << endl;
#endif
			_confidence.push_back(0);
			continue;
		}


		//create a trackedobject to get various statistics
		//including area and x,y,z position of the goal
		ObjectType goal_actual(_contours[i], "Actual Goal", 0);
		TrackedObject goal_tracked_obj(0, _goal_shape, br, depth_z_max, _fov_size, _frame_size, -((float)_camera_angle/10.) * M_PI / 180.0);
		//TrackedObject goal_tracked_obj(0, _goal_shape, br, depth_z_max, _fov_size, _frame_size, -16 * M_PI / 180.0);

		// Gets the bounding box area observed divided by the
		// bounding box area calculated given goal size and distance
		// For an object the size of a goal we'd expect this to be
		// close to 1.0 with some variance due to perspective
		float exp_area = goal_tracked_obj.getScreenPosition(_fov_size, _frame_size).area();
		float actualScreenArea = (float)br.area() / exp_area;

		if (((exp_area / br.area()) < 0.20) || ((exp_area / br.area()) > 5.00))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " area out of range for depth (depth_min/depth_max/act/exp/ratio):" << depth_z_min << "/" << depth_z_max << "/" << br.area() << "/" << exp_area << "/" << actualScreenArea << endl;
#endif
			_confidence.push_back(0);
			continue;
		}
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
		float confidence_height      = createConfidence(_goal_height, 0.4, goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0);
		float confidence_com_x       = createConfidence(com_percent_expected.x, 0.125,  com_percent_actual.x);
		float confidence_com_y       = createConfidence(com_percent_expected.y, 0.1539207,  com_percent_actual.y);
		float confidence_filled_area = createConfidence(filledPercentageExpected, 0.33,   filledPercentageActual);
		float confidence_ratio       = createConfidence(expectedRatio, 0.3,  actualRatio);
		float confidence_screen_area = createConfidence(1.0, 0.75,  actualScreenArea);

		// higher is better
		float confidence = (confidence_height + confidence_com_x + confidence_com_y + confidence_filled_area + confidence_ratio/2. + confidence_screen_area/2.) / 5.0;
		_confidence.push_back(confidence);

#ifdef VERBOSE
		cout << "-------------------------------------------" << endl;
		cout << "Contour " << i << endl;
		cout << "confidence_height: " << confidence_height << endl;
		cout << "confidence_com_x: " << confidence_com_x << endl;
		cout << "confidence_com_y: " << confidence_com_y << endl;
		cout << "confidence_filled_area: " << confidence_filled_area << endl;
		cout << "confidence_ratio: " << confidence_ratio << endl;
		cout << "confidence_screen_area: " << confidence_screen_area << endl;
		cout << "confidence: " << confidence << endl;
		cout << "Height exp/act: " << _goal_height << "/" <<  goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0 << endl;
		cout << "Depth min/max: " << depth_z_min << "/" << depth_z_max << endl;
		cout << "Area exp/act: " << (int)exp_area << "/" << br.area() << endl;
		cout << "Aspect ratio exp/act : " << expectedRatio << "/" << actualRatio << endl;
		cout << "br.br().y: " << br.br().y << endl;
		cout << "-------------------------------------------" << endl;
#endif

		if (confidence > _min_valid_confidence)
		{
			// This goal passes the threshold required for us to consider it a goal
			// Add it to the list of best goals
			GoalInfo goal_info;

			goal_info.pos        = goal_tracked_obj.getPosition();
			goal_info.confidence = confidence;
			goal_info.distance   = hypotf(goal_info.pos.x, goal_info.pos.y);
			goal_info.angle 	 = atan2f(goal_info.pos.x, goal_info.pos.y) * 180. / M_PI;
			goal_info.rect   	 = br;

			best_goals.push_back(goal_info);
		}

		/*vector<string> info;
		info.push_back(to_string(confidence_height));
		info.push_back(to_string(confidence_com_x));
		info.push_back(to_string(confidence_com_y));
		info.push_back(to_string(confidence_filled_area));
		info.push_back(to_string(confidence_ratio));
		info.push_back(to_string(confidence));
		info.push_back(to_string(h_dist));
		info.push_back(to_string(goal_to_center_deg));
		info_writer.log(info); */
	}
#ifdef VERBOSE
	cout << best_goals.size() << " goals passed first detection" << endl;
#endif
	//at this point we have both the top and bottom pieces of tape recognized
	if(best_goals.size() > 0)
	{
		int best_index = 0;
		if (best_goals.size() > 1)
		{
			vector<bool> for_removal(best_goals.size());
			for_removal = { false };	
			//filter out the bottom tape
			for(int i = 0; i < best_goals.size(); i++) { //top
				for(int j = 0; j < best_goals.size(); j++) { //bottom
					//if i is above j
					if(abs((best_goals[i].pos.z - best_goals[j].pos.z) - 0.0508) < 0.01) //last number here is a tolerance
					if(best_goals[i].pos.y - best_goals[j].pos.y < 0.01) { //also a tolerance
						//mark this goal to be removed
						for_removal[j] = true;
#ifdef VERBOSE
						cout << "Marked a goal for removal" << endl;
#endif
					}
				}
			}
			for(int i = 0; i < best_goals.size(); i++) {
				if(!for_removal[i])
					best_goals_updated.push_back(best_goals[i]);
			}
#ifdef VERBOSE
			cout << "Marked " << best_goals.size() - best_goals_updated.size() << " goals for removal" << endl;
#endif
			//if there are multiple goals sort by confidence
			if(best_goals_updated.size() >= 2) {
				sort (best_goals_updated.begin(), best_goals_updated.end(), [ ] (const GoalInfo &lhs, const GoalInfo &rhs)
				{
					return lhs.confidence > rhs.confidence;
				});

			}
			
			best_index = 0;
			// Save a bunch of info about the goal
			_goal_pos      = best_goals_updated[best_index].pos;
			_dist_to_goal  = best_goals_updated[best_index].distance;
			_angle_to_goal = best_goals_updated[best_index].angle;
			_goal_rect     = best_goals_updated[best_index].rect;
		} else {
		
			// Save a bunch of info about the goal
			_goal_pos      = best_goals[best_index].pos;
			_dist_to_goal  = best_goals[best_index].distance;
			_angle_to_goal = best_goals[best_index].angle;
			_goal_rect     = best_goals[best_index].rect;
		
		}
		_pastRects.push_back(SmartRect(_goal_rect));
	}
	else
		_pastRects.push_back(SmartRect(Rect()));
	isValid();
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
float GoalDetector::distanceUsingFOV(const Rect &rect) const
{
	float percent_image = (float)rect.height / _frame_size.height;
	float size_fov = percent_image * _fov_size.y; //TODO fov size
	return _goal_shape.height() / (2.0 * tanf(size_fov / 2.0));
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
	return _isValid ? _goal_rect : Rect();
}

// Goal x,y,z position relative to robot
Point3f GoalDetector::goal_pos(void) const
{
	return _isValid ? _goal_pos : Point3f();
}

// Draw debugging info on frame - all non-filtered contours
// plus their confidence. Highlight the best bounding rect in
// a different color
void GoalDetector::drawOnFrame(Mat &image) const
{
	for (size_t i = 0; i < _contours.size(); i++)
	{
		drawContours(image, _contours, i, Scalar(0,0,255), 3);
		Rect br(boundingRect(_contours[i]));
		rectangle(image, br, Scalar(255,0,0), 2);
		stringstream confStr;
		confStr << fixed << setprecision(2) << _confidence[i];
		putText(image, confStr.str(), br.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(255,0,0));
		putText(image, to_string(i), br.br(), FONT_HERSHEY_PLAIN, 1, Scalar(0,255,0));
	}

	if(!(_pastRects[_pastRects.size() - 1] == SmartRect(Rect())))
		rectangle(image, _pastRects[_pastRects.size() - 1].myRect, Scalar(0,255,0), 2);
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
	for(auto it = _pastRects.begin() + 1; it != _pastRects.end(); ++it)
	{
		if(!(*it == currentRect))
		{
			_isValid = false;
			return;
		}
	}
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
