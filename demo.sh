#!/bin/bash
cd /home/ubuntu/2015VisionCode/
sudo ./set_freq.sh
if [ "$1" = "-video" ]
  then
	COUNTER=1 #loop counter (also can be number of first video)
	cd /home/ubuntu/bin_videos
	NUM_VIDEOS=$(ls -1 | wc -l) #number of videos in directory
	while [ true ]; do
	/home/ubuntu/2015VisionCode/bindetection/test --demo /home/ubuntu/bin_videos/video$(($COUNTER % $NUM_VIDEOS)).avi
	COUNTER=$((COUNTER + 1))
	done	
  else
	while [ true ]; do
	if [ $# = 1 ] 
	  then
		/home/ubuntu/2015VisionCode/bindetection/test --demo $1
	  else
		/home/ubuntu/2015VisionCode/bindetection/test --demo 0
	  fi
	done
  fi
