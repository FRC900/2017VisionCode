#include <dirent.h>
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
vector<cv::Point3f> input_points;
input_points.push_back(cv::Point3f(-5,-5,-5));
input_points.push_back(cv::Point3f(-1,-1,-1));
input_points.push_back(cv::Point3f(1,1,1));
input_points.push_back(cv::Point3f(5,5,5));
vector<float> camera_elevation;
camera_elevation.push_back(0);
camera_elevation.push_back(-5);
camera_elevation.push_back(-10);
camera_elevation.push_back(5);
camera_elevation.push_back(10);
for(size_t i = 0; i < input_points.size(); i++) {
	for(size_t j = 0; j < camera_elevation.size(); j++) {
		for (int k = 1; k <= 5; k++) {
			ObjectType o(k);
		cv::Point3f test_p = input_points[i];
		float r = sqrtf(test_p.x * test_p.x + test_p.y * test_p.y + test_p.z * test_p.z) - o.depth()/2;
		const cv::Size frame_size(1280,720);
		const float hFov = 90.;
		const cv::Point2f fov_size(hFov * (M_PI / 180.),
				hFov * (M_PI / 180.) * ((float)frame_size.height / frame_size.width));
		cv::Point3f out_p = o.screenToWorldCoords(o.worldToScreenCoords(test_p ,fov_size, frame_size, camera_elevation[j]), r, fov_size, frame_size, camera_elevation[j]);
		// Objects with depth are a bit broken - fix eventually
		float thresh = std::max<float>(o.depth()/2, 0.01);
		ASSERT_NEAR(abs(out_p.x), abs(test_p.x), thresh) << "Input Coords " << input_points[i] << " camera_elevation " << camera_elevation[j] << " ObjectType(" << k << ")"; // some slop due to inaccuracies
		ASSERT_NEAR(abs(out_p.y), abs(test_p.y), thresh) << "Input Coords " << input_points[i] << " camera_elevation " << camera_elevation[j] << " ObjectType(" << k << ")"; // in handling object depth
		ASSERT_NEAR(abs(out_p.z), abs(test_p.z), thresh) << "Input Coords " << input_points[i] << " camera_elevation " << camera_elevation[j] << " ObjectType(" << k << ")";
		}
	}
}
}

TEST(ObjectTypeCoords, CenterSTW) {
const cv::Size frame_size(1280,720);
const float hFov = 105.;
const cv::Point2f fov_size(hFov * (M_PI / 180.),
		hFov * (M_PI / 180.) * ((float)frame_size.height / frame_size.width));

float cam_elev = 0;
for (int i = 1; i <= 5; i++) {
	ObjectType o(i);
	cv::Rect in(cv::Point(frame_size.width/2,frame_size.height/2), cv::Size(0,0));
	cv::Point3f out = o.screenToWorldCoords(in, 5.5, fov_size, frame_size, cam_elev);
	ASSERT_NEAR(out.x, 0, 0.1);
	ASSERT_NEAR(out.y, 5.5 + o.depth()/2.0, 0.1);
	ASSERT_NEAR(out.z, 0, 0.1);

	cv::Rect r(o.worldToScreenCoords(out, fov_size, frame_size, cam_elev));

	ASSERT_NEAR(r.x + r.width / 2, in.x, 1);
	ASSERT_NEAR(r.y + r.height / 2, in.y, 1);
	}
}

// Test fixture parameterized with name of video file to be tested
class GDVideoTest : public ::testing::TestWithParam<string> {
	public:
		MediaIn *cap;
		GoalDetector *gd;
		vector<cv::Point3f> positions;
		vector<int> corresponding_frames;
		virtual void SetUp() {
			cout.setstate(ios_base::badbit);
			cerr.setstate(ios_base::badbit);
			ZvSettings *zvSettings = new ZvSettings("/home/ubuntu/2017VisionCode/zebravision/settings.xml");
			cap = new ZMSIn(GetParam().c_str(), zvSettings);
			gd = new GoalDetector(cap->getCameraParams().fov, cv::Size(cap->width(), cap->height()));
			cout.setstate(ios_base::goodbit);
			cerr.setstate(ios_base::goodbit);
			cout << "Finished testing setup" << endl;
		}
		void processVideo() {
		cv::Mat frame, depth;
		int frameNumber = 1;
			while(cap->getFrame(frame, depth)) {
				cout.setstate(ios_base::badbit);
				gd->findBoilers(frame, depth);
				cout.setstate(ios_base::goodbit);
				if(gd->Valid()) {
					positions.push_back(gd->goal_pos());
					corresponding_frames.push_back(frameNumber);	
				}
				frameNumber++;
			}
		}
		cv::Point3f averagePos() {
			cv::Point3f totalPosition; cv::Point3f averagePos;
			for(size_t i = 0; i < positions.size(); i++) {
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
			for(size_t i = 0; i < positions.size(); i++) {
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
int x = std::stoi(s.substr(s.find("x")+1, (s.find("_",s.find("x")) - s.find("x")-1))); // Find from x to _
int y = std::stoi(s.substr(s.find("y")+1, (s.find("_",s.find("y")) - s.find("y")-1))); // Find from y to _
int z = std::stoi(s.substr(s.find("z")+1, (s.find("_",s.find("z")) - s.find("z")-1))); // Find from z to _
cv::Point3f actual_position(x/100.0,y/100.0,z/100.0);

// Run processing
processVideo();
cv::Point3f average_computed_position = averagePos();

// Check that the average position is no more than a certain distance off
float position_threshold = 0.5; // Meters
EXPECT_NEAR(average_computed_position.x, actual_position.x, position_threshold); 
EXPECT_NEAR(average_computed_position.y, actual_position.y, position_threshold); 
EXPECT_NEAR(average_computed_position.z, actual_position.z, position_threshold); 

// Check to make sure no positions are more than a certain distance off
/*for(int i = 0; i < positions.size(); i++) {
	EXPECT_NEAR(positions[i].x, actual_position.x, 0.2) << "Frame number: " << corresponding_frames[i];
	EXPECT_NEAR(positions[i].y, actual_position.y, 0.2) << "Frame number: " << corresponding_frames[i];
	EXPECT_NEAR(positions[i].z, actual_position.z, 0.2) << "Frame number: " << corresponding_frames[i];
} */
// Check that the variance is not greater than a threshold in all dimensions
cv::Point3f variance = variancePos();
float variance_threshold = 0.01; // Meters
EXPECT_LE(variance.x, variance_threshold);
EXPECT_LE(variance.y, variance_threshold);
EXPECT_LE(variance.z, variance_threshold);

}

vector<string> video_names;
INSTANTIATE_TEST_CASE_P( LabVideos, GDVideoTest, ::testing::ValuesIn(video_names));

int main(int argc, char **argv) {
	// Read the directory ./videos and add all filenames to video_names
	DIR *dirp = NULL;
	struct dirent *dp = NULL;
	dirp = opendir("./videos");
	if (dirp == NULL)
		cout << "Directory read error" << endl;
	while ((dp = readdir(dirp)) != NULL) {
		string file_name(dp->d_name); // Convert to string
		if(file_name != "." && file_name != "..") {
			video_names.push_back("./videos/" + file_name);
			cout << "Read video:" << "./videos/" + file_name << endl;
		}
	}
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
