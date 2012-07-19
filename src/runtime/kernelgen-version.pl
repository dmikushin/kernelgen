#!/usr/bin/perl -w
my($rev) = `svn info | grep Revision`;
$rev =~ s/Revision:\s/r/;
$rev =~ s/\n//;
print "$rev";

