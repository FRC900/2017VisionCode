#include "gtest/gtest.h"
#include "objtype.hpp"
#include "GoalDetector.hpp"
#include "zmsin.hpp"

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
vector< cv::Point > input;
EXPECT_THROW(ObjectType(input, "test", 5.0), std::invalid_argument) << "Exception with empty contour";
input.push_back(cv::Point(0,0));
input.push_back(cv::Point(1,1));
input.push_back(cv::Point(1,0));

EXPECT_THROW(ObjectType(input, "test", -5.0), std::invalid_argument) << "Exception with negative depth";
ASSERT_EQ(ObjectType(input, "test", 5.0).shape().size(), 3) << "Contour copied correctly";
}

TEST(ObjectTypeConstructor, CustomContour2f) {
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
for(size_t i = 0; i < input_points.size(); i++) { 
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

cv::Rect r(o.worldToScreenCoords(out, fov_size, frame_size, cam_elev));

ASSERT_NEAR(r.x + r.width / 2, in.x, 0.001);
ASSERT_NEAR(r.y + r.height / 2, in.y, 0.001);
}

// Test fixture parameterized with name of video file to be tested
class GDVideoTest : public ::testing::TestWithParam<string> {
	public:
		MediaIn *cap;
		GoalDetector *gd;
		vector<cv::Point3f> positions;
		vector<int> corresponding_frames;
		virtual void SetUp() {
			ZvSettings *zvSettings = new ZvSettings("/home/ubuntu/2017VisionCode/zebravision/settings.xml");
			cap = new ZMSIn(GetParam().c_str(), zvSettings);
			gd = new GoalDetector(cap->getCameraParams().fov, cv::Size(cap->width(), cap->height()));
		}
		void processVideo() {
		cv::Mat frame, depth;
		int frameNumber = 1;
			while(!cap->getFrame(frame, depth)) {
				gd->findBoilers(frame, depth);
				if(gd->Valid()) {
					positions.push_back(gd->goal_pos());
					corresponding_frames.push_back(frameNumber);	
				}
				frameNumber++;
			}
		}
		cv::Point3f averagePos() {
			cv::Point3f totalPosition; cv::Point3f averagePos;
			for(int i = 0; i < positions.size(); i++) {
				totalPosition.x = positions[i].x + totalPosition.x;
				totalPosition.y = positions[i].y + totalPosition.y;
				totalPosition.z = positions[i].z + totalPosition.z;
			}
			averagePos.x = totalPosition.x / positions.size();
			averagePos.y = totalPosition.y / positions.size();
			averagePos.z = totalPosition.z / positions.size();
			return averagePos;
		}

		cv::Point3f variancePos() {
			cv::Point3f avg = averagePos();
			cv::Point3f totalVariance(0,0,0);
			for(int i = 0; i < positions.size(); i++) {
				totalVariance.x += abs( positions[i].x - avg.x);
				totalVariance.y += abs( positions[i].y - avg.y);
				totalVariance.z += abs( positions[i].z - avg.z);
			}
			cv::Point3f averageVariance;
			averageVariance.x = totalVariance.x / positions.size();
			averageVariance.y = totalVariance.y / positions.size();
			averageVariance.z = totalVariance.z / positions.size();
			return averageVariance;
		}
};

TEST_P(GDVideoTest, TestVideoFile) {
// Parse video file for information on the position
// File format is: utest_x[xcm]_y[ycm]_z[zcm].zms
string s = GetParam();
int x = std::stoi(s.substr(s.find("x"), s.find("x",s.find("_")))); // Find from x to _
int y = std::stoi(s.substr(s.find("y"), s.find("y",s.find("_")))); // Find from y to _
int z = std::stoi(s.substr(s.find("z"), s.find("z",s.find(".")))); // Find from z to .
cv::Point3f actual_position(x/100.0,y/100.0,z/100.0);

processVideo();
cv::Point3f average_computed_position = averagePos();

// Check that the average position is no more than a certain distance off
EXPECT_NEAR(average_computed_position.x, actual_position.x, 0.1); 
EXPECT_NEAR(average_computed_position.y, actual_position.y, 0.1); 
EXPECT_NEAR(average_computed_position.z, actual_position.z, 0.1); 

// Check to make sure no positions are more than a certain distance off
for(int i = 0; i < positions.size(); i++) {
	EXPECT_NEAR(positions[i].x, actual_position.x, 0.2) << "Frame number: " << corresponding_frames[i];
	EXPECT_NEAR(positions[i].y, actual_position.y, 0.2) << "Frame number: " << corresponding_frames[i];
	EXPECT_NEAR(positions[i].z, actual_position.z, 0.2) << "Frame number: " << corresponding_frames[i];
}

}
// Fill in names of video files here
INSTANTIATE_TEST_CASE_P( LabVideos, GDVideoTest, ::testing::Values("videos/utest_x510_y101_z535.zms","videos/utest_x510_y101_z535.zms"));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
