#!/bin/bash
# Usage : create_cascade.sh <cascade directory>
# This script looks in a given cascade directory.  For each stage it finds, generate a full cascade output .xml
# file for that stage. Generate both "old" and "new" formats since both are needed depending on which
# type of classifier is being run (Haar vs LBP) and whether it is on the CPU or GPU.
# For the oldformat, apply a fix to the file format to work around a bug in OpenCV

# When this script is done, there will be a number of cascade_xx.xml files. Each of them can be loaded 
# as the classifier info for cascade classifier testing. Higher numbers in the <xx> mean more stages of 
# training : better rejection of false positives along with slightly lower detection rate of real targets.
# Our code allows the user to dynamically switch between various stages to see how the detection quality
# changes with the number of stages.

i=0

while [ -f $1/stage$i.xml ];
do
   i=$((i + 1))
  opencv_traincascade -data $1 -vec positives.vec -bg negatives.dat -w 20 -h 20 -numStages $i -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 7000 -numNeg 5000 -mode ALL -precalcValBufSize 2000 -precalcIdxBufSize 2000
  mv $1/cascade.xml $1/cascade_$i.xml
  opencv_traincascade -data $1 -vec positives.vec -bg negatives.dat -w 20 -h 20 -numStages $i -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 7000 -numNeg 5000 -mode ALL -precalcValBufSize 2000 -precalcIdxBufSize 2000 -baseFormatSave
  cat $1/cascade.xml | sed 's?</cascade>?</output>?' | sed 's?<cascade>?<output type_id=\"opencv-haar-classifier\">?' > $1/cascade_oldformat_$i.xml
done
