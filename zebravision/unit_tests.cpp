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
/*TEST(ObjectTypeWTS, BadInputs) {
ObjectType o(1);

}*/


int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
