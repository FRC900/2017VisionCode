#include <iostream>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Int32.h>

class RobotDriver
{
	private:
		//! The node handle we'll be using
		ros::NodeHandle nh_;
		//! We will be publishing to the "/base_controller/command" topic to issue commands
		ros::Publisher cmd_vel_pub_;
		ros::Publisher test_pub_;

	public:
		//! ROS node initialization
			//set up the publisher for the cmd_vel topic
		RobotDriver(const ros::NodeHandle &nh) :
			nh_(nh),
			cmd_vel_pub_(nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1)),
			test_pub_(nh_.advertise<std_msgs::Int32>("test_pub", 1))
		{
		}

		//! Loop forever while sending drive commands based on keyboard input
		bool driveKeyboard(void)
		{
			std::cout << "Type a command and then press enter.  "
				"Use '+' to move forward, 'l' to turn left, "
				"'r' to turn right, '.' to exit.\n";

			//we will be sending commands of type "twist"
			geometry_msgs::Twist base_cmd;
			std_msgs::Int32 int32_msg;

			char cmd[50];
			bool quit = false;
			int index = 0;
			while(nh_.ok() && !quit){

				std::cin.getline(cmd, 50);
#if 0
				if(cmd[0]!='+' && cmd[0]!='l' && cmd[0]!='r' && cmd[0]!='.')
				{
					std::cout << "unknown command:" << cmd << "\n";
					continue;
				}
#endif

				base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;
				//move forward
				if(cmd[0]=='+'){
					base_cmd.linear.x = 0.25;
				}
				//turn left (yaw) and drive forward at the same time
				else if(cmd[0]=='l'){
					base_cmd.angular.z = 0.75;
					base_cmd.linear.x = 0.25;
				}
				//turn right (yaw) and drive forward at the same time
				else if(cmd[0]=='r'){
					base_cmd.angular.z = -0.75;
					base_cmd.linear.x = 0.25;
				}
				//quit
				else if(cmd[0]=='.'){
					quit = true;
				}
				else // e-stop like right now
				{
				}

				//publish the assembled command
				cmd_vel_pub_.publish(base_cmd);
				int32_msg.data = index++;
				test_pub_.publish(int32_msg);
			}
			return true;
		}
};

int main(int argc, char** argv)
{
	//init the ROS node
	ros::init(argc, argv, "robot_driver");
	ros::NodeHandle nh;

	RobotDriver driver(nh);
	driver.driveKeyboard();
}
