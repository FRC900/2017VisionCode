FROM ros:kinetic-robot

# Copy code to /ws
ADD zebROS_ws/src /ws/zebROS_ws/src
ADD zebravision/ /ws/zebravision
ADD navXTimeSync/ /ws/navXTimeSync
ADD Docker_setup_ROS.sh /

# Run script
RUN bash /Docker_setup_ROS.sh

# Delete source code? If you really want a tiny image do this
# RUN rm -rf /ws/zebravision /ws/zebROS_ws/src /ws/navXTimeSync

# Clear APT lists to keep size down
#RUN rm -rf /var/lib/apt/lists/
