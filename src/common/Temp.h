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
