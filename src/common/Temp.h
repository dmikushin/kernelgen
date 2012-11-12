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

#ifndef TEMP_H
#define TEMP_H

#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <malloc.h>
#include <string>
#include <sstream>

namespace kernelgen { namespace utils {

	class TempFile
	{
		friend class Temp;

		std::string name;
		int fd;

		llvm::tool_output_file& file;

		TempFile(std::string& name, int fd, llvm::tool_output_file& file) :
			name(name), fd(fd), file(file) { }

	public :

		int getFD() const { return fd; }
		const std::string& getName() const { return name; }

		void keep() { file.keep(); }

		// Dump content into temp file.
		void download(const char* content, size_t size)
		{
			using namespace std;
			fstream stream;
			stream.open(name.c_str(),
				fstream::binary | fstream::out | fstream::trunc);
			stream.write(content, size);
			stream.close();
		}

		// Upload content from temp file.
		void upload(char** content, size_t* size)
		{
			using namespace std;
			ifstream stream(name.c_str(), ios::in | ios::binary);
			filebuf* buffer = stream.rdbuf();

			// Get file size and load its data.
			*size = buffer->pubseekoff(0, ios::end, ios::in);
			buffer->pubseekpos(0, ios::in);
			*content = (char*)malloc(*size + 1);
			buffer->sgetn(*content, *size);
			(*content)[*size] = '\0';

			stream.close();
		}
	};

	class Temp
	{
	public :
		static TempFile getFile(std::string mask = "%%%%%%%%", bool closefd = true);
	};

} // namespace utils
} // namespace kernelgen

#endif // TEMP_H
