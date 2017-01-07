#!/usr/bin/perl
#
use Cwd 'abs_path';
my $cwd = abs_path(".");

opendir(my $dh, ".") || die "Can not open $basedir : $!";
@zv_dirs = grep {-d } readdir($dh);
closedir $dh;
my @d12_model_dirs = grep { /d12_/  } @zv_dirs;
my @d24_model_dirs = ("d24_1");

my @videodirs = ("/home/kjaget/ball_videos/test", "/home/kjaget/ball_videos/20160210");
my @videos = ();
for $videodir (@videodirs)
{
	opendir(my $dh, $videodir) || die "Can not open $videodir : $!";
	my @this_videos = ();
	push @this_videos, grep { /.avi$/ && -f "$videodir/$_" } readdir($dh);
	for $video (sort @this_videos)
	{
		push @videos, $videodir . "/" . $video;
	}
	closedir $dh;
}

for $video (sort @videos)
{
	print "$video,,,,";
}
print "\n\nD12 Net,D24 Net, Detected, Actual, False Positives, Frames\n";

for $d24_dir (sort @d24_model_dirs)
{
	my $d24_num = -1;
	if ($d24_dir =~ /d24_(\d+)/)
	{
		$d24_num = $1;
	}
	next if ($d24_num == -1);

	for $d12_dir (sort @d12_model_dirs)
	{
		my $d12_num = -1;
		if ($d12_dir =~ /d12_(\d+)/)
		{
			$d12_num = $1;
		}
		next if ($d12_num == -1);
		print "$d12_num, $d24_num, ";
		for $video (sort @videos)
		{
			open (my $pipeh, "./zv --d12Dir=$d12_num --d24Dir=$d24_num --batch --groundTruth \"$videodir/$video\"  2>/dev/null|");
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
		print "\n";
	}
}
