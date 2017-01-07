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

my $d12_dir = 99;
opendir(my $dh, "d12_".$d12_dir) || die "Can not open d12_$d12_dir : $!";
my @this_d12_stages = ();
push @this_d12_stages, grep { /.caffemodel$/ } readdir($dh);
closedir $dh;

for $snapshot (@this_d12_stages)
{
	if ($snapshot =~ /snapshot_iter_(\d+).caffemodel/)
	{
		push @d12_stages, $1;
	}
}
my $d24_dir = 99;
opendir(my $dh, "d24_".$d24_dir) || die "Can not open d24_$d24_dir : $!";
my @this_d24_stages = ();
push @this_d24_stages, grep { /.caffemodel$/ } readdir($dh);
closedir $dh;

for $snapshot (@this_d24_stages)
{
	if ($snapshot =~ /snapshot_iter_(\d+).caffemodel/)
	{
		push @d24_stages, int($1);
	}
}

print "\n\nD12 Net, D12 Epoch, D24 Net, D24 Epoch, Detected, Actual, False Positives, Frames\n";

for $d24_stage (sort {$a <=> $b} @d24_stages)
{
	for $d12_stage (sort {$a <=> $b} @d12_stages)
	{
		#next if ($d12_stage < 296112);
		print "$d12_dir, $d12_stage, $d24_dir, $d24_stage, ";
		for $video (sort @videos)
		{
			open (my $pipeh, "./zv --d12Dir=$d12_dir --d12Stage=$d12_stage --d24Dir=$d24_dir --d24Stage=$d24_stage --batch --groundTruth \"$videodir/$video\"  2>/dev/null|");
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
