#!/usr/bin/perl
#
use Cwd 'abs_path';
my $cwd = abs_path(".");

my $videodir = "/home/kjaget/ball_videos/test";
my $basedir = "/home/kjaget/test";
my $level = "d12";

opendir(my $dh, $basedir) || die "Can not open $basedir : $!";
my @model_dirs = grep { -d "$basedir/$_" } readdir($dh);
closedir $dh;

opendir(my $dh, $videodir) || die "Can not open $videodir : $!";
my @videos = grep { /.avi$/ && -f "$videodir/$_" } readdir($dh);
closedir $dh;

for $video (sort @videos)
{
	print "$video\n";
}

for $dir (sort @model_dirs)
{
	if (-f "$basedir/$dir/deploy.prototxt")
	{
		print "$dir\n";
	}
}

for $dir (sort @model_dirs)
{
	my $fulldir = $basedir."/".$dir;
	#print "$fulldir ";
	if (-f "$fulldir/deploy.prototxt")
	{
		open (my $fh, "$fulldir/train_val.prototxt") || die "Could not open $fulldir/train_val.prototxt : $!";
		while (my $line = <$fh>)
		{
			if ($line =~ /mean_file: "(.+)\/mean.binaryproto/)
			{
				#print "Mean dir = $1\n";
				$mean_dir = $1;
			}
			elsif ($line =~ /type: "Convolution"/)
			{
				while (my $line = <$fh>)
				{
					if ($line =~ /num_output: (\d+)/)
					{
						$conv_out = $1;
						last;
					}
				}
			}
			elsif ($line =~ /type: "InnerProduct"/)
			{
				while (my $line = <$fh>)
				{
					if ($line =~ /num_output: (\d+)/)
					{
						$fc_out = $1;
						last;
					}
				}
				last;
			}
		}
		close ($fh);
		open (my $fh, "$fulldir/solver.prototxt") || die "Could not open $fulldir/solver.prototxt : $!";
		while (my $line = <$fh>)
		{
			if ($line =~ /base_lr: ([\d\.e-]+)/)
			{
				$base_lr = $1;
			}
		}
		close ($fh);
		`rm $level/*`;
		`cp $fulldir/* $level`;
		`cp $mean_dir/mean.binaryproto $level`;
		`cp $mean_dir/labels.txt $level`;
		opendir(my $dh, "$level") || die "Can not open $level: $!";
		my @snapshots = grep { /^snapshot_iter_\d+.caffemodel/ && -f "$level/$_" } readdir($dh);
		closedir $dh;
		for $snapshot (sort @snapshots)
		{
			if ($snapshot =~ /snapshot_iter_(\d+).caffemodel/)
			{
				print "$conv_out, $fc_out, $base_lr, $fulldir, $1, ";
				for $video (sort @videos)
				{
					open (my $pipeh, "./zv --batch --groundTruth --".$level."Stage=$1 $videodir/$video |");
					while ($line = <$pipeh>)
					{
						if ($line =~ /(\d+) of (\d+) ground truth objects/)
						{
							print "$1, $2, ";
						}
						elsif ($line =~ /(\d+) false positives found in (\d+) frames/)
						{
							print "$1, $2, ";
						}
					}
					close $pipeh;
				}
			}
			print "\n";
		}
	}
	else
	{
		print "\n";
	}

}

