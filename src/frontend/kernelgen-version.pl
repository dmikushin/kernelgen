#!/usr/bin/perl -w
my($rev) = `svn info | grep Revision`;
$rev =~ s/Revision:\s/r/;
$rev =~ s/\n//;
print "#include <stdio.h>\n";
print "int main(void) { printf(\"$rev\"); return 0; }\n";

