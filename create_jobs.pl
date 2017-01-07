#!/usr/bin/perl
#
# Script to iterate through net hyper-parameters
# Copy base job into ~/base_job - need train_val.prototxt, solver.prototxt and deploy.prototxt
# Run the script to create permutations of conv, fc1, fc2 output and learning rate
# Run the generated script to create outputs

my @conv_list = (4,6,8,10);
my @fc1_list = (4,6,8);
my @fc2_list = (2);
my @lr_list = (0.001, 0.0005, 0.0001);

open (my $scriptfh, ">", "runcaffe.sh") or die ("Can not open runcaffe.sh for writing : $!");
for my $conv_val (@conv_list)
{
	for my $fc1_val (@fc1_list)
	{
		for my $fc2_val (@fc2_list)
		{
			for my $lr_val (@lr_list)
			{
				my $outdir = $conv_val."c_".$fc1_val."fc1_".$fc2_val."fc2_".$lr_val."LR";
				print $scriptfh "cd $outdir\n";
				print $scriptfh "/home/ubuntu/caffe-nv/build/install/bin/caffe train -solver solver.prototxt > caffe_output.log\n";
				print $scriptfh "cd ..\n";
				mkdir $outdir;
				open (my $fh, "<", "/home/kjaget/base_job/train_val.prototxt") or die ("Can not open ~/base_job/train_val.prototxt : $!");
				open (my $outfh, ">", $outdir."/train_val.prototxt");
				while ($line = <$fh>)
				{
					if ($line =~ /type: "Convolution"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $conv_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					elsif ($line =~ /name: "fc1_d12"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $fc1_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					elsif ($line =~ /name: "fc2_d12"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $fc2_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					else 
					{
						print $outfh $line;
					}
				}
				close ($fh);
				close ($outfh);

				open (my $fh, "<", "/home/kjaget/base_job/deploy.prototxt") or die ("Can not open ~/base_job/deploy.prototxt : $!");
				open (my $outfh, ">", $outdir."/deploy.prototxt");
				while ($line = <$fh>)
				{
					if ($line =~ /type: "Convolution"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $conv_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					elsif ($line =~ /name: "fc1_d12"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $fc1_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					elsif ($line =~ /name: "fc2_d12"/)
					{
						print $outfh $line;
						while ($line = <$fh>)
						{
							if ($line =~ /num_output: \d+/)
							{
								print $outfh "    num_output: $fc2_val\n";
								last;
							}
							else
							{
								print $outfh $line;
							}
						}
					}
					else 
					{
						print $outfh $line;
					}
				}
				close ($fh);
				close ($outfh);
				open (my $fh, "<", "/home/kjaget/base_job/solver.prototxt") or die ("Can not open ~/base_job/train_val.prototxt : $!");
				open my $outfh, ">", $outdir."/solver.prototxt";
				while ($line = <$fh>)
				{
					if ($line =~ /base_lr: /)
					{
						print $outfh "base_lr: $lr_val\n";
					}
					else 
					{
						print $outfh $line;
					}
				}
				close ($fh);
				close ($outfh);
			}
		}
	}
}
close ($scriptfh);
