#include <iostream>

#include <ros/ros.h>

#include "encoder_example/stampedUInt16.h"

using namespace std;

// Dummy class to encapsulate encoder access
// This might not be necessary depending on how
// the HAL implements CAN-based encoder reads?
class EncoderHardwareIF
{
	public:
		EncoderHardwareIF(const char *name)
		{
			(void)name;
		}
		// Return a dummy value for testing
		unsigned short Read(void) const
		{
			static unsigned short val;
			val += 11;
			return val;
		}
};


int main(int argc, char** argv)
{
	ros::init(argc, argv, "encoder_example");

	ros::NodeHandle nh;

	// Set up publishers - data is a standard ROS header (timestamp, frame, seq_number)
	// plus the encoder value;
	ros::Publisher pub = nh.advertise<encoder_example::stampedUInt16>("/encoder", 50);

	encoder_example::stampedUInt16 stampedEncoderData;
	stampedEncoderData.header.frame_id = "encoder_frame"; // make me a parameter?

	ros::Rate loop_time(50); // run at 50Hz

	EncoderHardwareIF encHW("foo"); // TODO : initialize interface for reading encoder data

	while(ros::ok()) 
	{
		//set the timestamp for all headers
		stampedEncoderData.header.stamp = ros::Time::now();
		stampedEncoderData.data = encHW.Read();

		// Need to decide exactly what to publish.  For this simple
		// example it is just a raw encoder value. For a swerve drive
		// wheel we might need velocity in X and Y directions.  For 
		// a shooter we might just report velocity.  For linear actuators
		// maybe publish a 1-d position
		//
		// Or do we have other topics which read the raw encoder values
		// and convert them into real-world units of position / velocity?
		//
		//publish to ROS topics
		pub.publish(stampedEncoderData); // queue up data to publish
		ros::spinOnce();        // yield and publish data
		loop_time.sleep();      // sleep until next update period hits
	}

	return 0;
}
