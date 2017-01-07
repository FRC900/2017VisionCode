#! /bin/bash

# Run this once to initialize a cascade training run.  
# Outputs two files :
# - posivites.vec : this is a collection of positive images to look for. 
#                   View them using opencv_createsamples -w 20 -h 20 -vec positives.vec
# - negatives.dat : a text file with a list of images which DO NOT contain
#                   the objects being detectec.
# Positive samples should be in the positive_images subdir.  They can be any size (the
# script will resize the actual input to training to a uniform size) and could be color
# or B&W.  They should all be the same aspect ratio, and that aspect ratio should
# match the -w and -h arguments to various commands in this script (the size doens't have to match,
# just the aspect ratio i.e. if -w and -h are 20, all input images should be square but they
# don't all have to be exactly 20x20).  Ideally the images would show the target from a 
# variety of angles (including elevation changes), lighting conditions, against various
# backgrounds, etc.  The more unique conditions included here, the better the detection accuracy.
# The negative_images subdir should contain images which *DO NOT* have the object
# being detected.  These can be any size, aspect ratio, just so long as there are 
# absolutely no examples of the object to detect are present at any size.  Be especially
# sure this is true when grabbing "hard negatives" - false positives caputured from
# testing intermediate stages - since sometimes the classifer will grab an image
# which has the object included in a larger detected image.
#
# Note that additional negatives can be added at any time during the training process,
# but adding positives will probably not work well.
#
# Things to edit :
# - number of samples created on the "perl createtrainsamples.pl" run. More is relatively better, 
# but more also increases trainging time.  Finding the right balance isn't an exact science.

# Create list of negative images, randomize the list
/usr/bin/find negative_images -name \*.png > negatives.dat
/usr/bin/find negative_images -name \*.jpg >> negatives.dat
shuf negatives.dat > temp.dat
mv temp.dat negatives.dat

# Create list of positive images
#/usr/bin/find positive_images -name \*.png > positives.dat
#/usr/bin/find positive_images -name \*.jpg >> positives.dat


# For each positive image, create a number of randomly rotated versions of that image
# This creates a .vec file for each positive input image, each containing multiple images 
# rotated random amounts
# KCJ - perl createtrainsamples.pl positives.dat negatives.dat . 12000 | tee foo.txt

# Merge each set of randomized versions of the images into one big .vec file
rm positives.vec
# KCJ - /usr/bin/find . -name \*.vec > vectors.dat
# KCJ - mergevec/src/mergevec.exe vectors.dat ordered_positives.vec

# Randomize the order of those images inside the .vec file.
# If you change w/h in the first line, change the numbers in the next two 
# lines to match
# KCJ - mergevec/src/vec2img ordered_positives.vec samples%04d.png -w 20 -h 20 | shuf > info.dat
# KCJ - sed 's/$/ 1 0 0 20 20/' info.dat > random_info.dat

#find /home/kjaget/CNN/ball/ -name \*.png | xargs identify -format '\"%d/%f\" 1 0 0 %w %h\n' > info.dat
shuf -n 25000 info.dat > random_info.dat
opencv_createsamples -info random_info.dat -vec positives.vec -num `wc -l random_info.dat` -w 24 -h 24
rm random_info.dat 

