FROM ros:kinetic-robot
# install ros tutorials packages
RUN apt update && apt install -y \
    && rm -rf /var/lib/apt/lists/
RUN mkdir /ws
ADD . /ws
