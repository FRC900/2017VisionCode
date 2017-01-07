#! /usr/bin/perl

# Script to optimze the performance of early stages of a classifier.
# It repeatedly runs the first few stages of tarining on a random
# subset of negative input samples. For each training run, test the
# performance by running he generated classifier on a short test
# video. Since the classifier changes depending on the negative images
# used, the performance of the generated classifier varies depending
# on the quantity of filtering needed to weed out each random choice
# of negative samples.
#
# The script saves the best performing of the trained classifiers.
# Once the user is happy that they've found a quick enough preliminary
# classifier, this saved best classifier is trained with however
# many additional stages are needed.
#
my $best_fps = 0;

# Read previous best FPS value. Allows restarting script
# without losing past results.
if (open(FPS_FILE, "classifier_bin_save/best_fps.txt"))
{
   $line = <FPS_FILE>;
   $best_fps = $1 if ($line =~ /([0-9]*\.?[0-9]+)/);
   close FPS_FILE;
}
print "Previous best FPS = $best_fps\n";

# This basically duplicates the work done in prep.sh
# Generate list of positive images once since it never changes
readpipe("/usr/bin/find positive_images -name \*.png > positives.dat");
while (1)
{
   # Grab a list of negative images and randomize their order
   print "Creating negatives\n";
   readpipe("/usr/bin/find negative_images -name \*.png > negatives.dat");
   readpipe("shuf negatives.dat > temp.dat");
   readpipe("mv temp.dat negatives.dat");

   # Each positive sample is converted into multiple images by
   # rotating it random ways.  These are compiled into one .vec file
   # containing all of the processed images.  Randomize the order of the
   # positive samples inside this file.
   print "Creating sample .vecs\n";
   readpipe("rm *.vec");
   readpipe("perl createtrainsamples.pl positives.dat negatives.dat . 12000"); 
   readpipe("/usr/bin/find . -name \\*.vec > vectors.dat");
   readpipe("mergevec/src/mergevec.exe vectors.dat ordered_positives.vec");
   readpipe("mergevec/src/vec2img ordered_positives.vec samples%04d.png -w 20 -h 20 | shuf > info.dat");
   readpipe('sed \'s/$/ 1 0 0 20 20/\' info.dat > random_info.dat');

   print "Creating positives.vec\n";
   readpipe( "mergevec/src/createsamples -info random_info.dat -vec positives.vec -num \`wc -l random_info.dat\` -w 20 -h 20");
   readpipe ("rm samples????.png samples[0-9]????.png");
   readpipe("mkdir -p classifier_bin_x");
   readpipe("rm -f classifier_bin_x/*");

   # Train 15 stages of the classifier. This is quick enough so we don't
   # waste too much time training but still get a reasonbly performing classifier
   # TODO : randomize numNegative value each time through in a range of something 
   #        like 25-300% of numPos
   print "Running classifier training\n";
   readpipe ("opencv_traincascade -data classifier_bin_x -vec positives.vec -bg negatives.dat -w 20 -h 20 -numStages 15 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 11250 -numNeg 7000 -featureType LBP -precalcValBufSize 1750 -precalcIdxBufSize 1750 -maxWeakCount 1000");

   # Run classifier on sample video, record the speed
   print "Testing detection speed\n";
   open (FPSTEST, "../bindetection/fpstest |");
   $fps = <FPSTEST>;
   chomp $fps;
   close (FPSTEST);
   print "$fps FPS\n";
   if ($fps > $best_fps)
   {
      # If this is the best performing classifer, save it
      $best_fps = $fps;
      print "New best FPS\n";
      readpipe("mkdir -p classifier_bin_save");
      readpipe("rm -f classifier_bin_save/*");
      readpipe("cp classifier_bin_x/* classifier_bin_save");
      readpipe("mv negatives.dat classifier_bin_save");
      readpipe("mv positives.vec classifier_bin_save/positives.vec~");
      readpipe("echo $best_fps > classifier_bin_save/best_fps.txt");
   }
}
