#include "gtest/gtest.h"
#include "objtype.hpp"

/*class objTypeFixture: public ::testing::test { 
public:
   objTypeFixture() {
	o = new ObjectType(1);
}
   void SetUp( ) { 
       // code here will execute just before the test ensues 
   
	}
 
   void TearDown( ) { 
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }
 
   ~objTypeFixture( )  { 
       // cleanup any pending stuff, but no exceptions allowed
   }
 
   ObjectType o;
};*/

TEST(ObjectTypeTest, NamePresent) {
ObjectType o(1);
EXPECT_EQ(o.name(),"ball");

}

int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
