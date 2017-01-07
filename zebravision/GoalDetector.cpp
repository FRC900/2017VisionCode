#include <iomanip>
#include <opencv2/highgui/highgui.hpp>
#include "GoalDetector.hpp"

using namespace std;
using namespace cv;

//#define VERBOSE

GoalDetector::GoalDetector(cv::Point2f fov_size, cv::Size frame_size, bool gui) :
	_goal_shape(3),
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

			
		if(_general){
		//create a transform that will transform from the skewed shape to the
		//non skewed shape
		Point2f input_points[4];
		Point2f output_points[4];
			//create a rotatedrect surrounding the contour and input the points into an array
			RotatedRect warped_shape = minAreaRect(_contours[i]);
			warped_shape.points(input_points);

			//find how far slanted the goal is away from the screen
			Mat contour_mask = Mat::zeros(depth.size(), CV_8UC1);
			drawContours(contour_mask,_contours,i,Scalar(255), CV_FILLED);
			std::pair<double,double> slope_of_shape = utils::slopeOfMasked(ObjectType(1), depth, contour_mask, _fov_size);
			float x_angle = atan(slope_of_shape.first);
			float y_angle = atan(slope_of_shape.second);
			cout << "Contour  " << i << " angle of goal away from screen: " << x_angle * (180/M_PI) << " " << y_angle * (180/M_PI) << endl;
			//create another rotatedrect to transform the points into. This is in the same spot
			//as the actual contour but the size is changed to match what it would be if facing it straight on
			RotatedRect unwarped_shape(warped_shape.center, Size(cos(x_angle)*warped_shape.size.width, cos(y_angle)*warped_shape.size.height), 0);

			//something about opencv's handling of rotatedrects causes the contours to spin 90 if the width < height
			//if(unwarped_shape.size.width < unwarped_shape.size.height)
			//	unwarped_shape.angle = -90;

			unwarped_shape.points(output_points);

			//create a transformation that maps points from warped to unwarped goal
			Mat warp_transform(3,3,CV_32FC1);
			warp_transform = getPerspectiveTransform(input_points, output_points);

			//apply the transformation to the contour
			//in order to do this the points to be transformed have to be floats
			vector<Point2f> unwarped_contour_f;
			vector<Point> unwarped_contour;
			for(size_t j = 0; j < _contours[i].size(); j++)
				unwarped_contour_f.push_back(Point2f((float)((_contours[i])[j]).x,(float)((_contours[i])[j]).y));

			cv::perspectiveTransform(unwarped_contour_f,unwarped_contour_f, warp_transform);

			//convert back from floats and apply to contours list
			for(size_t j = 0; j < unwarped_contour_f.size(); j++)
	    			_contours[i][j] = Point((int)unwarped_contour_f[j].x,(int)unwarped_contour_f[j].y);
		}

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

		// Remove objects too low on the screen - these can't
		// be goals. Should stop the robot from admiring its
		// reflection in the diamond-plate at the end of the field
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

		// Since the goal is a U shape, there should be bright pixels
		// at the bottom center of the contour and dimmer ones in the
		// middle going towards the top. Check for that here
		Mat topMidCol(threshold_image(Rect(cvRound(br.tl().x + br.width * .35f), br.tl().y, cvRound(br.width * .3f), cvRound(br.height * .4f))));
		Mat botMidCol(threshold_image(Rect(br.tl().x, cvRound(br.tl().y + br.height * .85f), br.width, cvRound(br.height * .15f))));
		double topMaxCol;
		minMaxLoc(topMidCol, NULL, &topMaxCol);
		double botMaxCol;
		minMaxLoc(botMidCol, NULL, &botMaxCol);
		// The max pixel value in the bottom rows of the
		// middle column should be > 0 and the max pixel
		// value in the top rows of that same column should be 0
		if ((topMaxCol >= 1.0) || (botMaxCol < 1.0))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " middle column values wrong (top/bot):" << (int)topMaxCol << "/" << (int)botMaxCol << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		// Grab max pixel values along a line through
		// the middle row of the bounding rect. There
		// should be high values on the left and
		// right and no high values in the middle
		// Sample the edges at both .35 and .65 of the way
		// down the rect to find goals which look
		// angled due to their offset
		Mat leftTopMidRow(threshold_image(Rect(br.tl().x, cvRound(br.tl().y + br.height * .35f), cvRound(br.width * .15f), 1)));
		Mat leftBotMidRow(threshold_image(Rect(br.tl().x, cvRound(br.tl().y + br.height * .65f), cvRound(br.width * .15f), 1)));
		Mat rightTopMidRow(threshold_image(Rect(br.tl().x + cvRound(br.width * .85f), cvRound(br.tl().y + br.height * .35f), cvRound(br.width * .15f), 1)));
		Mat rightBotMidRow(threshold_image(Rect(br.tl().x + cvRound(br.width * .85f), cvRound(br.tl().y + br.height * .65f), cvRound(br.width * .15f), 1)));
		Mat centerMidRow(threshold_image(Rect(br.tl().x + cvRound(br.width * 3.f / 8.f), cvRound(br.tl().y + br.height / 4.f), cvRound(br.width / 4.f), 1)));
		double dummy;
		double rightMaxRow;
		minMaxLoc(rightTopMidRow, NULL, &dummy);
		minMaxLoc(rightBotMidRow, NULL, &rightMaxRow);
		rightMaxRow = max(dummy, rightMaxRow);
		double leftMaxRow;
		minMaxLoc(leftTopMidRow, NULL, &dummy);
		minMaxLoc(leftBotMidRow, NULL, &leftMaxRow);
		leftMaxRow = max(dummy, leftMaxRow);
		double centerMaxRow;
		minMaxLoc(centerMidRow, NULL, &centerMaxRow);
		if ((leftMaxRow < 1.0) || (centerMaxRow > 0.) || (rightMaxRow < 1.0))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " middle row wrong (left / center / right):" << (int)leftMaxRow << "/" <<(int)centerMaxRow << "/" << (int)rightMaxRow << endl;
			cout << "\tRight(x2): " << rightTopMidRow << "/" << rightBotMidRow << " Center: " << centerMidRow << " Left(x2): " << leftTopMidRow << "/" << leftBotMidRow << endl;
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
		if ((actualRatio <= (1.0/2.25)) || (actualRatio >= 2.25))
		{
#ifdef VERBOSE
			cout << "Contour " << i << " aspectRatio out of range:" << actualRatio << endl;
#endif
			_confidence.push_back(0);
			continue;
		}

		//parameters for the normal distributions
		//values for standard deviation were determined by
		//taking the standard deviation of a bunch of values from the goal
		//confidence is near 0.5 when value is near the mean
		//confidence is small or large when value is not near mean
		float confidence_height      = createConfidence(_goal_height, 0.4, goal_tracked_obj.getPosition().z - _goal_shape.height() / 2.0);
		float confidence_com_x       = createConfidence(com_percent_expected.x, 0.125,  com_percent_actual.x);
		float confidence_com_y       = createConfidence(com_percent_expected.y, 0.1539207,  com_percent_actual.y);
		float confidence_filled_area = createConfidence(filledPercentageExpected, 0.33,   filledPercentageActual);
		float confidence_ratio       = createConfidence(expectedRatio, 0.537392,  actualRatio);
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
	if(best_goals.size() > 0)
	{
		int best_index = 0;
		if (best_goals.size() > 1)
		{
#ifdef VERBOSE
			cout << best_goals.size() << " goals passed first detection" << endl;
#endif
			// Remove down to 3 goals based on confidence
			// Sort by decreasing confidence - first entries will
			// have the highest confidence
			sort (best_goals.begin(), best_goals.end(), [ ] (const GoalInfo &lhs, const GoalInfo &rhs)
			{
				return lhs.confidence > rhs.confidence;
			});

			// Sort the top 3 entries by width
			sort (best_goals.begin(), min(best_goals.end(), best_goals.begin()+3), [ ] (const GoalInfo &lhs, const GoalInfo &rhs)
			{
				return lhs.rect.width > rhs.rect.width;
			});
			//decide between final 2 goals based on either width or position on screen
			//decide how to decide based on if the goals have extremely similar widths
			//note that goals near the edge of the screen will appear wider
			//due to camera distortions
			if(abs(best_goals[0].rect.width - best_goals[1].rect.width) > 5)
			{
#ifdef VERBOSE
				cout << "Deciding based on width" << endl;
#endif
				if(best_goals[0].rect.width > best_goals[1].rect.width)
					best_index = 0;
				else
					best_index = 1;
			}
			else
			{
#ifdef VERBOSE
				cout << "Deciding based on position " << endl;
#endif
				if(best_goals[0].rect.br().x < best_goals[1].rect.br().x)
					best_index = 1;
				else
					best_index = 0;
			}
		}

		// Save a bunch of info about the goal
		_goal_pos      = best_goals[best_index].pos;
		_dist_to_goal  = best_goals[best_index].distance;
		_angle_to_goal = best_goals[best_index].angle;
		_goal_rect     = best_goals[best_index].rect;

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
