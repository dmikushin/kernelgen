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

#ifndef VERBOSE_H
#define VERBOSE_H

#include <iostream>
#include <vector>

#include <llvm/Support/raw_ostream.h>

namespace kernelgen {

	class Verbose : public llvm::raw_fd_ostream
	{
		int mode, filter, always;

		virtual void write_impl(const char *Ptr, size_t Size)
		{
			if ((mode & filter) || always)
				llvm::outs().write(Ptr, Size);
		}

	public :

		Verbose(int mode = 0) : mode(mode), filter(1), always(0), llvm::raw_fd_ostream(STDOUT_FILENO, false, true) { }

		struct Color
		{
			const int value;
			Color(int value) : value(value) { }
		};

		static Color Black;
		static Color Red;
		static Color Green;
		static Color Yellow;
		static Color Blue;
		static Color Magenta;
		static Color Cyan;
		static Color White;
		static Color SavedColor;
		static Color Reset;

		struct Mode
		{
			const int value;
			Mode(int value) : value(value) { }

		    int operator&(Mode& other)
		    {
		    	return this->value & other.value;
		    }

		    bool operator==(Mode& other)
			{
		    	return this->value == other.value;
			}

		    bool operator!=(Mode& other)
			{
		    	return this->value != other.value;
			}
		};

		inline Mode getMode() const { return Mode(mode); }
		inline void setMode(Mode mode) { this->mode = mode.value; }

		static Mode Always;
		static Mode Default;
		static Mode Disable;
		static Mode Summary;
		static Mode Sources;
		static Mode ISA;
		static Mode DataIO;
		static Mode Hostcall;
		static Mode Polly;
		static Mode Perf;
		static Mode Alloca;

		struct Action
		{
			int code;

			Action(int code) : code(code) { }
		};

		static Action Flush;

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Verbose::Color& color)
		{
			if (color.value == Reset.value)
				OS.resetColor();
			else
				OS.changeColor((raw_ostream::Colors)color.value);
			return OS;
		}

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Verbose::Mode& filter)
		{
			Verbose* verbose = static_cast<Verbose*>(&OS);
			if (filter.value == Verbose::Always.value)
				verbose->always = 1;
			else if (filter.value == Verbose::Default.value)
			{
				verbose->filter = verbose->mode;
				verbose->always = 0;
			}
			else
			{
				verbose->filter = filter.value;
				verbose->always = 0;
			}
			return OS;
		}

		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, const Verbose::Action& action)
		{
			if (action.code == Verbose::Flush.code)
				llvm::outs().flush();
			return OS;
		}

		// Output command line from the given vector of arguments.
		friend llvm::raw_ostream& operator<<(llvm::raw_ostream& OS, std::vector<const char*>& args)
		{
			for (int i = 0, ie = args.size(); i != ie; i++)
				if (args[i]) OS << args[i] << " ";
			OS << "\n";
			return OS;
		}
	};

} // namespace kernelgen

#endif // VERBOSE_H
