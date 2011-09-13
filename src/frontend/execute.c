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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define READ	0
#define WRITE	1
#define ERROR	2

#define SZBUF	1024

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
		close(p_stdin[WRITE]);
		dup2(p_stdin[READ], READ);
		close(p_stdout[READ]);
		dup2(p_stdout[WRITE], WRITE);
		close(p_stderr[READ]);
		dup2(p_stderr[WRITE], ERROR);
		execl("/bin/sh", "sh", "-c", command, NULL);
		exit(0);
	}

	if (!infp)
		close(p_stdin[WRITE]);
	else
		*infp = p_stdin[WRITE];

	if (!outfp)
		close(p_stdout[READ]);
	else
		*outfp = p_stdout[READ];
	
	if (!errfp)
		close(p_stderr[READ]);
	else
		*errfp = p_stderr[READ];

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
int execute(const char* command, char* in, size_t szin,
	char** out, size_t* szout, char** err, size_t* szerr)
{
	int infp, outfp, errfp;
	if (run(command, &infp, &outfp, &errfp) <= 0)
	{
		fprintf(stderr, "Cannot run command %s\n", command);
		return 1;
	}
	
	// Write to input stream.
	if (in)	write(infp, in, szin);
	close(infp);

	// Wait for child process to finish.
	int status;
	if (wait(&status) == -1)
	{
		fprintf(stderr, "Cannot synchronize with child process\n");
		return 1;
	}

	// Read error stream.
	if (err && (multiplex(errfp) > 0))
	{
		int length = 0, capacity = 0;
		*err = NULL;
		do
		{
			capacity += SZBUF;
			*err = realloc(*err, capacity);
			int szread = read(errfp, *err + capacity - SZBUF, SZBUF);
			*szerr += szread;
		}
		while (*szerr == capacity);
		close(errfp);
	}

	// Read output stream.
	if (out && (multiplex(outfp) > 0))
	{
		int capacity = 0;
		*out = NULL;
		do
		{
			capacity += SZBUF;
			*out = realloc(*out, capacity);
			int szread = read(outfp, *out + capacity - SZBUF, SZBUF);
			*szout += szread;
		}
		while (*szout == capacity);
		close(outfp);
	}
	
	return 0;
}

