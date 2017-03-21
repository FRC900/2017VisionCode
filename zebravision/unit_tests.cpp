#include "gtest/gtest.h"
#include "objtype.hpp"

using namespace std;


TEST(ObjectTypeConstructor, Id) {
for(int i = 1; i < 6; i++) {
	ObjectType o(i);
	ASSERT_GT(o.name().length(), 0) << "Type: " << i << " name defined";
	ASSERT_GT(o.shape().size(), 0) << "Type: " << i << "contour defined";
	ASSERT_GE(o.real_height(), 0) << "Type: " << i << "Real world height are defined";
	}
}
TEST(ObjectTypeConstructor, CustomContour) {

vector< cv::Point2f > input;
EXPECT_THROW(ObjectType(input, "test", 5.0), std::invalid_argument) << "Exception with empty contour";
input.push_back(cv::Point2f(0,0));
input.push_back(cv::Point2f(1,1));
input.push_back(cv::Point2f(1,0));

EXPECT_THROW(ObjectType(input, "test", -5.0), std::invalid_argument) << "Exception with negative depth";
ASSERT_EQ(ObjectType(input, "test", 5.0).shape().size(), 3) << "Contour copied correctly";

}

//WorldToScreen
TEST(ObjectTypeCoords, Reversible) {

ObjectType o(1);
vector<cv::Point3f> input_points;
input_points.push_back(cv::Point3f(-5,-5,-5));
input_points.push_back(cv::Point3f(0,0,0));
input_points.push_back(cv::Point3f(5,5,5));
for(int i = 0; i < input_points.size(); i++) { 
	cv::Point3f test_p = input_points[i];
	float r = sqrtf(test_p.x * test_p.x + test_p.y * test_p.y + test_p.z * test_p.z);
	cv::Point fov_size(90.0 / (2 * M_PI),  90.0 / (2 * M_PI) * (9. / 16.));
	cv::Size frame_size(1080,720);
	float cam_elev = 0;
	cv::Point3f out_p = o.screenToWorldCoords(o.worldToScreenCoords(test_p,fov_size,frame_size, cam_elev), r, fov_size, frame_size, cam_elev);
	ASSERT_NEAR(abs(out_p.x), abs(test_p.x), 0.2);
	ASSERT_NEAR(abs(out_p.y), abs(test_p.y), 0.2);
	ASSERT_NEAR(abs(out_p.z), abs(test_p.z), 0.2);
}
}

TEST(ObjectTypeCoords, CenterSTW) {
const cv::Size frame_size(1280,720);
const float hFov = 105.;
const cv::Point fov_size(hFov * (M_PI / 180.),
		hFov * (M_PI / 180.) * ((float)frame_size.height / frame_size.width));

float cam_elev = 0;
ObjectType o(1);
cv::Rect in(cv::Point(frame_size.width/2,frame_size.height/2), cv::Size(0,0));
cv::Point3f out = o.screenToWorldCoords(in, 5.5, fov_size, frame_size, cam_elev);
ASSERT_NEAR(out.x, 0, 0.1);
ASSERT_NEAR(out.y, 5.5 +o.depth()/2.0, 0.1);
ASSERT_NEAR(out.z, 0, 0.1);

}



int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
