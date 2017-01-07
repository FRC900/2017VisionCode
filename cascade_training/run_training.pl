#!/usr/bin/perl

# Main script to run training. Make sure prep.sh was run previously.  Also make sure
# the -data dir exists.
while (1)
{
	# traincascade command line
	# Options to edit
	# -data classifier_dir_name : output dir for training data. Make sure thie exists before training
	# -numStages : the max number of classifier stages to generate. Note that the training can be stopped
	#              and restarted at will. The output of each stage of training is saved, so 
	#              restarting the training will resume by restarting the incomplete stage from the
	#              beginning.  The create_cascade script will post-process these intermediate files
	#              to create full .xml classifier files which can be loaded by the detection
	#              code. This way intermediate stages of training can be used for several purposes :
	#                     - run it in detection code against test input to see if the
	#                       detection using the current number of stages is good enough.
	#                       Since it is possible to test intermediate stages, we typically
	#                       train more stages than needed and do an evaluation of which
	#                       intermediate stage is good enough for our use. The balance of
	#                       rejecting enough false positives without rejecting too many
	#                       instances of the target is hard to guess before-hand. It is easier
	#                       to see when running against test video.
	#                     - Swithcing between stages on the fly gives an idea of how
	#                       the detector improves with each stage
	#                     - running detection code and saving false positives. These are 
	#                       images which are detected but aren't the object we're looking for.  
	#                       Feeding these false positives back into detection as one of the 
	#                       negative samples will help the classifier learn to reject the
	#                       false positives in later stages of training.
	# -numPos : number of positive samples to use in each stage. This is 
	#          a bit less than the number of images in the positive_images,
	#          since some of those images are used for other parts of training.
	#          Set this to 80-90% of the number of images created by prep.sh
	# -numNeg : number of negative images to use. A numNeg value on the order of numPos is 
	#           reasonable. This isn't the number of images in negative_samples - the classifier 
	#           training takes each e.g. 20x20 window of each negative image as a negative
	#           sample. Thus a few large images will produce lots of possible negative
	#           samples. On the other hand, each stage of training only picks those
	#           samples which aren't filtered out by all the previous stages. Once you
	#           get a few stages in, lots of these samples will be filtered out.
	#           The training code wil complain if there aren't enough unfiltered
	#           negative samples left to hit the numNeg value. In that case, generate
	#           more negative samples, either by grabbing random images which don't have
	#           the positive sample in them or by running the detection code and grabbing
	#           false positives (so-called hard negatives).
	# -w/-h : width and height of sample to train on.  Somewhere in the range of 20x20 seems optimal.
	#         Note that the actual detection code starts by looking for samples this small and then
	#         scales up until it hits a selectable max. This means that it will never detect objects
	#         smaller than wxh, but will detect ones larger.
	# BufSizes : these are precalculated buffers, sized in MB. Make sure you have enough memory to 
	#            hold these buffers. 
	# maxWeakCount : number of iterations to process for each stage.  Ideally the clssifier will
	#                iterate until the number of positives detected is <minHitRate> and the number
	#                of false positives is less than <maxFalseAlarmRate>. This means that each stage will
	#                detect 99.9% of positives that are there, but also generate 50% false positives.  
	#                maxWeakCount is the number of iterations the max classifier training iterations
	#                run to get to these values. If this value is too low, the classifier training will bail
	#                out early - most likely this will mean that there will be more than 50% false positives
	#                let through.  Since the code exits when the other conditions are satisified there's
	#                not much downside to setting this really high. Most classifier stages will finish
	#                way before this iteration limit is hit.
	my $pid = open(PIPE, "opencv_traincascade -data classifier_ball_1 -vec positives.vec -bg negatives.dat -w 24 -h 24 -numStages 55 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 22000 -numNeg 15000 -featureType LBP -precalcValBufSize 3400 -precalcIdxBufSize 3400 -maxWeakCount 1000 |");
	while ($line = <PIPE>)
	{
		print $line;
		# Output is | 5 | 0.999 | 0.567
		# 5 is iteration number
		# 0.999 is ratio of positive samples correctly detected
		# 0.567 is the number of false positives
		if ($line =~ /\|\s+(\d+)\|\s+([0-9\.]+)\|\s+([0-9\.]+\|)/)
		{
			if ($1 eq "5")
			{
				# re-randomize the negative sample list after five iterations
				# so that they are different for the next stage
				# Note that this will also include new negative images added while
				# the run is in progress (but won't pick up new positives
				# since that really won't work well anyway).
				`bash redo_negatives.sh`;
			}
			# Sometimes the training doesn't make progress.  There are two cases 
			# we commonly see :
			#  1. 100% of positive samples is detected after a reasonable number of iterations.
			#  You'd think this is good, but in good training runs this typically drops to 0.999xxx 
			#  as the number of false positives goes down. If the ratio stays at 1, it seems to 
			#  mean that the training can't find a way to distinguish between real and false
			#  positives.
			#  2. False postive rate stays high even after lots of iterations.  This is bad
			#  for the same basic reason as #1 - the code can't figure out how to weed out
			#  the false positives from the real ones.
			# If the training is stuck, kill it and restart the stage. Since the negative list will be
			# re-randomized the training will be different each time and does eventually work
			if ((($1 > 25) && ($2 eq "1")) || 
				(($1 > 75) && ($3 > 0.99)))
			{
				print "Killing\n";
				kill $pid;
				close (PIPE);
			}
		}
	}
}
