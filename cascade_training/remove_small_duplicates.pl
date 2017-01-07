#!/bin/perl
#
# Remove small versions of pictures if the larger
# versions are no longer around.  One of the tools used to
# remove dupes doesn't handle smaller images, so only the larger
# ones are detected.  Use this script to get the smaller copies 
# of those images as well.
#

opendir my $dh, "." || die $!;

while (readdir $dh)
{
   $file_exists{$_} = 1;
}
closedir $dh;

foreach $file_name (keys %file_exists)
{
   if ($file_name =~ /(.+)_s.png$/)
   {
      if (!defined($file_exists{$1.".png"}))
      {
	 $file_remove{$file_name} = 1;
      }
   }
}

foreach $file_name (sort keys %file_remove)
{
   `rm \"$file_name\"`;
}

