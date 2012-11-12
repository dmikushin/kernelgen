#ifndef VERBOSE_H
#define VERBOSE_H

#include "runtime.h"

#include <iostream>
#include <vector>

namespace kernelgen { namespace utils {

	class Verbose
	{
	public :

		// Dump the command, if verbose mode is enabled.
		static void cmd(std::vector<const char*> args)
		{
			using namespace std;
			for (vector<const char*>::iterator i = args.begin(), ie = args.end(); i != ie; i++)
				if (*i) cout << *i << " ";
			cout << endl;
		}
	};

} // namespace utils
} // namespace kernelgen

#endif // VERBOSE_H
