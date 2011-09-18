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

#include "util.h"

#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define READ	0
#define WRITE	1
#define ERROR	2

#define SZBUF	1024

using namespace std;

static pid_t run(const char *command, int* infp, int* outfp, int* errfp)
{
	int p_stdin[2], p_stdout[2], p_stderr[2];
	pid_t pid;
	
	if (pipe(p_stdin))
	{
		fprintf(stderr, "Cannot setup pipe for stdin\n");
		return 1;
	}
	if (pipe(p_stdout))
	{
		fprintf(stderr, "Cannot setup pipe for stdout\n");
		return 1;
	}
	if (pipe(p_stderr))
	{
		fprintf(stderr, "Cannot setup pipe for stderr\n");
		return 1;
	}

	pid = fork();
	if (pid < 0)
		return pid;
	else if (pid == 0)
	{
		if (infp)
		{
			close(p_stdin[WRITE]);
			dup2(p_stdin[READ], READ);
		}
		if (outfp)
		{
			close(p_stdout[READ]);
			dup2(p_stdout[WRITE], WRITE);
		}
		if (errfp)
		{
			close(p_stderr[READ]);
			dup2(p_stderr[WRITE], ERROR);
		}
		if (execl("/bin/sh", "sh", "-c", command, NULL) == -1)
			exit(-1);
		exit(0);
	}

	if (infp) *infp = p_stdin[WRITE];
	if (outfp) *outfp = p_stdout[READ];
	if (errfp) *errfp = p_stderr[READ];

	return pid;
}

static int multiplex(int fd)
{
	fd_set fdset;
	FD_ZERO(&fdset);
	FD_SET(fd, &fdset);

	struct timeval tv;
	tv.tv_sec = 0;
	tv.tv_usec = 0;
	
	int retval = select(fd + 1, &fdset, NULL, NULL, &tv);
	if (retval == -1)
	{
		fprintf(stderr, "Error while multiplexing fd = %d\n", fd);
		return -1;
	}
	return retval;
}

// Execute the specified command in the system shell, supplying
// input stream content and returning results from output and
// error streams.
int execute(string command, list<string> args,
	string in, string* out, string* err)
{
	string cmd = command;
	for (list<string>::iterator it = args.begin(); it != args.end(); it++)
		cmd += " " + *it;

	int infp, outfp, errfp;
	if (run(cmd.c_str(), in.size() ? &infp : NULL,
		out ? &outfp : NULL, err ? &errfp : NULL) <= 0)
	{
		cerr << "Cannot run command " << command << endl;
		return 1;
	}
	
	// Write to input stream.
	if (in.size()) write(infp, in.c_str(), in.size());
	close(infp);

	// Wait for child process to finish.
	int status;
	if (wait(&status) == -1)
	{
		cerr << "Cannot synchronize with child process" << endl;
		return 1;
	}
	if (status) return status;

	// Read error stream.
	if (err && (multiplex(errfp) > 0))
	{
		int length = 0, capacity = 0;
		char* cerr = NULL;
		do
		{
			capacity += SZBUF;
			cerr = (char*)realloc((void*)cerr, capacity);
			int szread = read(errfp, cerr + capacity - SZBUF, SZBUF);
			length += szread;
		}
		while (length == capacity);
		close(errfp);
		err->assign(cerr, length);
	}

	// Read output stream.
	if (out && (multiplex(outfp) > 0))
	{
		int length = 0, capacity = 0;
		char* cout = NULL;
		do
		{
			capacity += SZBUF;
			cout = (char*)realloc((void*)cout, capacity);
			int szread = read(outfp, cout + capacity - SZBUF, SZBUF);
			length += szread;
		}
		while (length == capacity);
		close(outfp);
		out->assign(cout, length);
	}
	
	return 0;
}

