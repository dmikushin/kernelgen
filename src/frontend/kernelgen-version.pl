#!/usr/bin/perl -w
#!/usr/bin/perl -w
use List::Util qw[min max];

my($result) = 0;
my(@dirs) = split("\n", `find ../.. -type d -not -iwholename '*.svn*'`);
foreach $dir (@dirs)
{
	my($rev) = "";
	$rev = `svn info $dir 2>/dev/null | grep Revision`;
	if ($rev ne "")
	{
		$rev =~ s/Revision:\s//;
		$rev =~ s/\n//;
		if ($rev =~ m/\d+/)
		{
			 $result = max($result, $rev);
		}
	}
}
print "#include <stdio.h>\n";
print "int main(void) { printf(\"r$result\"); return 0; }\n";

