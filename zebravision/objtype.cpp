// A class for defining objects we're trying to 
// detect.  The class stores information about shape 
// and size of the objects in real-world measurements
#include "objtype.hpp"

using namespace std;
using namespace cv;

ObjectType::ObjectType(int contour_type_id=1) {
	switch(contour_type_id) {
		//loads one of the preset shapes into the
		//object

		case 1: //a ball!
			depth_ = 0.2476; // meters
			contour_.push_back(Point2f(0,0));
			contour_.push_back(Point2f(0, depth_));
			contour_.push_back(Point2f(depth_, depth_));
			contour_.push_back(Point2f(depth_,0));
			name_="ball";
			break;

		case 2: //a bin (just because)
			depth_ = 0.5588;
			contour_.push_back(Point2f(0,0));
			contour_.push_back(Point2f(0,0.5842));
			contour_.push_back(Point2f(0.5842,0.5842));
			contour_.push_back(Point2f(0.5842,0));
			name_="bin";
			break;

		case 3: //2016 Goal
			{
				depth_ = 0;
				float max_y = .3048;
				contour_.push_back(Point2f(0, max_y - 0));
				contour_.push_back(Point2f(0, max_y - 0.3048));
				contour_.push_back(Point2f(0.0508, max_y - 0.3048));
				contour_.push_back(Point2f(0.0508, max_y - 0.0508));
				contour_.push_back(Point2f(0.508-0.0508, max_y - 0.0508));
				contour_.push_back(Point2f(0.508-0.0508, max_y - 0.3048));
				contour_.push_back(Point2f(0.508, max_y - 0.3048));
				contour_.push_back(Point2f(0.508, max_y - 0));
				name_="goal";
			}
			break;
		case 4: //top piece of tape (2017)
			contour_.push_back(Point2f(0,0));
			contour_.push_back(Point2f(0, 0.1010));
			contour_.push_back(Point2f(0.381, 0.1010));
			contour_.push_back(Point2f(0.381, 0));
		case 5: //bottom piece of tape (2017)
			contour_.push_back(Point2f(0,0));
			contour_.push_back(Point2f(0, 0.1010/2.0));
			contour_.push_back(Point2f(0.381, 0.1010/2.0));
			contour_.push_back(Point2f(0.381, 0));
		default:
			cerr << "error initializing object!" << endl;
	}

	computeProperties();

}

ObjectType::ObjectType(const vector< Point2f > &contour_in, const string &name_in, const float &depth_in) :
	contour_(contour_in),
	depth_(depth_in),
	name_(name_in)	
{
	computeProperties();
}

ObjectType::ObjectType(const vector< Point > &contour_in, const string &name_in, const float &depth_in):
	depth_(depth_in),
	name_(name_in)
{
	for(size_t i = 0; i < contour_in.size(); i++)
	{
		Point2f p;
		p.x = (float)contour_in[i].x;
		p.y = (float)contour_in[i].y;
		contour_.push_back(p);
	}
	computeProperties();

}

void ObjectType::computeProperties()
{
	float min_x = numeric_limits<float>::max();
	float min_y = numeric_limits<float>::max();
	float max_x = numeric_limits<float>::min();
	float max_y = numeric_limits<float>::min();
	for (auto it = contour_.cbegin(); it != contour_.cend(); ++it)
	{
		min_x = min(min_x, it->x);
		min_y = min(min_y, it->y);
		max_x = max(max_x, it->x);
		max_y = max(max_y, it->y);
	}
	width_ = max_x - min_x;
	height_ = max_y - min_y;
	area_ = contourArea(contour_);

	//compute moments and use them to find center of mass
	Moments mu = moments(contour_, false);
	com_ = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
}

Point3f ObjectType::screenToWorldCoords(const Rect &screen_position, double avg_depth, const Point2f &fov_size, const Size &frame_size, float cameraElevation) const
{
	/*
	Method:
		find the center of the rect
		compute the distance from the center of the rect to center of image (pixels)
		convert to degrees based on fov and image size
		do a polar to cartesian cordinate conversion to find x,y,z of object
	Equations:
		x=rsin(inclination) * cos(azimuth)
		y=rsin(inclination) * sin(azimuth)
		z=rcos(inclination)
	Notes:
		Z is up, X is left-right, and Y is forward
		(0,0,0) = (r,0,0) = right in front of you
	*/

	// TODO : see about using camera params cx and cy here
	// Those will be the actual optical center of the frame
	Point2f rect_center(
			screen_position.x + (screen_position.width  / 2.0),
			screen_position.y + (screen_position.height / 2.0));
	Point2f dist_to_center(
			rect_center.x - (frame_size.width / 2.0),
			-rect_center.y + (frame_size.height / 2.0));

// This uses formula from http://www.chiefdelphi.com/forums/showpost.php?p=1571187&postcount=4
	float azimuth = atanf(dist_to_center.x / (.5 * frame_size.width / tanf(fov_size.x / 2)));
	float inclination = atanf(dist_to_center.y / (.5 * frame_size.height / tanf(fov_size.y / 2))) - cameraElevation;

	// avg_depth is to front of object.  Add in half the
	// object's depth to move to the center of it
	avg_depth += depth_ / 2.;
	Point3f retPt(
			avg_depth * cosf(inclination) * sinf(azimuth),
			avg_depth * cosf(inclination) * cosf(azimuth),
			avg_depth * sinf(inclination));

	//cout << "Distance to center: " << dist_to_center << endl;
	//cout << "Actual Inclination: " << inclination << endl;
	//cout << "Actual Azimuth: " << azimuth << endl;
	//cout << "Actual location: " << retPt << endl;
	return retPt;
}

Rect ObjectType::worldToScreenCoords(const Point3f &_position, const Point2f &fov_size, const Size &frame_size, float cameraElevation) const
{
	float r = sqrtf(_position.x * _position.x + _position.y * _position.y + _position.z * _position.z) - depth_ / 2.;
	float azimuth = asinf(_position.x / sqrt(_position.x * _position.x + _position.y * _position.y));
	float inclination = asinf( _position.z / r ) + cameraElevation;

	//inverse of formula in screenToWorldCoords()
	Point2f dist_to_center(
			tanf(azimuth) * (0.5 * frame_size.width / tanf(fov_size.x / 2)),
			tanf(inclination) * (0.5 * frame_size.height / tanf(fov_size.y / 2)));
	
	//cout << "Distance to center: " << dist_to_center << endl;
	Point2f rect_center(
			dist_to_center.x + (frame_size.width / 2.0),
			-dist_to_center.y + (frame_size.height / 2.0));

	Point2f angular_size(2.0 * atan2f(width_, 2.0*r), 2.0 * atan2f(height_, 2.0*r));
	Point2f screen_size(
			angular_size.x * (frame_size.width / fov_size.x),
			angular_size.y * (frame_size.height / fov_size.y));

	Point topLeft(
			cvRound(rect_center.x - (screen_size.x / 2.0)),
			cvRound(rect_center.y - (screen_size.y / 2.0)));

	return Rect(topLeft.x, topLeft.y, cvRound(screen_size.x), cvRound(screen_size.y));
}

float ObjectType::expectedDepth(const Rect &screen_position, const Size &frame_size, const float hfov) const
{
	// TODO : use larger of width, height for slightly better resolution
	float percent_image = (float)screen_position.width / frame_size.width;
	float size_fov      = percent_image * hfov;
	return width_ / (2.0 * tanf(size_fov / 2.0)) - depth_ / 2.;
}

bool ObjectType::operator== (const ObjectType &t1) const 
{
	return this->shape() == t1.shape();
}

