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

	public:
		//! ROS node initialization
			//set up the publisher for the cmd_vel topic
		RobotDriver(const ros::NodeHandle &nh) :
			nh_(nh),
			cmd_vel_pub_(nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1))
		{
		}

		//! Loop forever while sending drive commands based on keyboard input
		bool driveKeyboard(void)
		{
			std::cout << "Type a command and then press enter.  "
				"Use wasd to strafe, 'l' to turn left, "
				"'r' to turn right, '.' to exit.\n";

			//we will be sending commands of type "twist"
			geometry_msgs::Twist base_cmd;
			std_msgs::Int32 int32_msg;

			const float scale = 0.25;

			char cmd[50];
			bool quit = false;
			while(nh_.ok() && !quit)
			{
				std::cin.getline(cmd, 50);

				base_cmd.linear.x = base_cmd.linear.y = base_cmd.angular.z = 0;
				switch (cmd[0])
				{
					case '+' :
					case 'w' :
						base_cmd.linear.x = scale;
						break;

					case 's' :
						base_cmd.linear.x = -scale;
						break;

					case 'a':
						base_cmd.linear.y = -scale;
						break;

					case 'd':
						base_cmd.linear.y = scale;
						break;

					case 'l':
						base_cmd.angular.z = 3 * scale;
						base_cmd.linear.x = scale;
						break;

					case 'r':
						base_cmd.angular.z = -3 * scale;
						base_cmd.linear.x = scale;
						break;

					case '.':
						quit = true;

					default: // any other key is e-stop
						break;

				}

				//publish the assembled command
				cmd_vel_pub_.publish(base_cmd);
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
