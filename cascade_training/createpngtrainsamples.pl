#!/usr/bin/perl
use File::Basename;
use strict;
##########################################################################
# Create samples from an image applying distortions repeatedly 
# (create many many samples from many images applying distortions)
#
#  perl createtrainsamples.pl <positives.dat> <negatives.dat> <png_output_dir>
#      [<totalnum = 7000>] [<createsample_command_options = ./createsamples -w 20 -h 20...>]
#  ex) perl createtrainsamples.pl positives.dat negatives.dat samples
#
# Author: Naotoshi Seo
# Expanded by FRC Team 900 for neural net training
#########################################################################
#
# Create a positives.dat file containing names of all of the input images
# Create a similar negatives.dat file holding names of the negatives
#
# white :  my $cmd = '../opencv_createsamples_color/opencv_createsamples_color -bgcolor 0x511efc -bgthresh 0x511e03 -maxxangle .2 -maxyangle .2 -maxzangle 6.283 -maxidev 40 -w 24 -h 24 -hsv';
# blue : my $cmd = '../opencv_createsamples_color/opencv_createsamples_color -bgcolor 0x73ee64 -bgthresh 0x061433 -maxxangle .2 -maxyangle .2 -maxzangle 6.283 -maxidev 40 -w 24 -h 24 -hsv';
# purple : my $cmd = '../opencv_createsamples_color/opencv_createsamples_color -bgcolor 0x96c997 -bgthresh 0x143667 -maxxangle .2 -maxyangle .2 -maxzangle 6.283 -maxidev 40 -w 24 -h 24 -hsv';
my $cmd = '/home/ubuntu/2016VisionCode/opencv_createsamples_color/opencv_createsamples_color -bgcolor 0 -bgthresh 0 -maxxangle .2 -maxyangle .2 -maxzangle 6.283 -maxidev 40 -w 24 -h 24 -hsv';
my $totalnum = 7000;
my $tmpfile  = 'tmp';

if ($#ARGV < 2) {
    print "Usage: perl createpngtrainsamples.pl\n";
    print "  <positives_collection_filename>\n";
    print "  <negatives_collection_filename>\n";
    print "  <output_dirname>\n";
    print "  [<totalnum = " . $totalnum . ">]\n";
    print "  [<createsample_command_options = '" . $cmd . "'>]\n";
    exit;
}
my $positive  = $ARGV[0];
my $negative  = $ARGV[1];
my $outputdir = $ARGV[2];
$totalnum     = $ARGV[3] if ($#ARGV > 2);
$cmd          = $ARGV[4] if ($#ARGV > 3);

open(POSITIVE, "< $positive");
my @positives = <POSITIVE>;
close(POSITIVE);

open(NEGATIVE, "< $negative");
my @negatives = <NEGATIVE>;
close(NEGATIVE);

# number of generated images from one image so that total will be $totalnum
my $numfloor  = int($totalnum / ($#positives+1));
my $numremain = $totalnum - $numfloor * ($#positives+1);

for (my $k = 0; $k <= $#positives; $k++ ) {
    my $img = $positives[$k];
    my $num = ($k < $numremain) ? $numfloor + 1 : $numfloor;

    # Pick up negative images randomly
    my @localnegatives = ();
    for (my $i = 0; $i < $num; $i++) {
        my $ind = int(rand($#negatives+1));
        push(@localnegatives, $negatives[$ind]);
    }
    open(TMP, "> $tmpfile");
    print TMP @localnegatives;
    close(TMP);
	#system("cat $tmpfile");

    !chomp($img);
    my $imgdirlen = length(dirname($img));
	my $pngfn = $outputdir . substr($img, $imgdirlen) . "_chroma_%5.5d.png" ;
    print "$cmd -img \"$img\" -bg $tmpfile -pngfnformat \"$pngfn\" -num $num" . "\n";
    system("$cmd -img \"$img\" -bg $tmpfile -pngfnformat \"$pngfn\" -num $num");
}
unlink($tmpfile);

