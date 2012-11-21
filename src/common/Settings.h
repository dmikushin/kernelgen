/*
 * KernelGen - the LLVM-based compiler with GPU kernels generation over C backend.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it freely,
 * subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented;
 * you must not claim that you wrote the original software.
 * If you use this software in a product, an acknowledgment
 * in the product documentation would be appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such,
 * and must not be misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>

namespace kernelgen {

	class Settings
	{
		// Kernels runmode (target).
		int runmode;

		// Verbose output.
		int verbose;

		// Debug mode.
		int debug;

		// Subarchitecture setting, e.g. "sm_20" for sm_20 CUDA target.
		// This setting overrides the value returned by device CC query
		// (if less or equal).
		std::string subarch;

	public :

		inline int getRunmode() const { return runmode; }
		inline int getVerbose() const { return verbose; }
		inline int getDebug() const { return debug; }
		inline std::string getSubarch() const { return subarch; }

		Settings();
	};

} // kernelgen

#endif // SETTINGS_H
