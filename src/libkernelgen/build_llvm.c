/*
 * KGen - the LLVM-based compiler with GPU kernels generation over C backend.
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

#include "kernelgen_int.h"

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

static pid_t execute(const char *command, int* infp, int* outfp, int* errfp)
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

kernelgen_status_t kernelgen_build_llvm(
	struct kernelgen_launch_config_t* l, const char* options,
	char** target_source, size_t* target_source_size)
{
	// Being quiet optimistic initially...
	kernelgen_status_t result;
	result.value = kernelgen_success;
	result.runmode = l->runmode;
	
	int iplatform = kernelgen_thread_platform_index;
	int idevice = kernelgen_thread_device_index;
	
	// If kernel was previously compiled for entire
	// platform/device, return now.
	if ((l->config->last_platform_index == iplatform) &&
		(l->config->last_device_index == idevice))
		return result;
	
	// Otherwise, optimize kernel IR and build it.
	// TODO: optimize IR.
	
	// Create a separate process with shell script
	// executing llc tool. Redirect its input/output
	// file descriptors to supply source code and
	// retrieve generated object code from entire process.
	const char* command_fmt = "llc -march=c | gcc %s -E -o- -xc -";
	int command_length = snprintf(NULL, 0, command_fmt, options);
	if (command_length < 0)
	{
		result.value = kernelgen_error_compilation_failed;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot construct invocation command\n");
		return result;
	}
	char* command = (char*)malloc(command_length + 1);
	sprintf(command, command_fmt, options);
	int infp, outfp, errfp;
	if (execute(command, &infp, &outfp, &errfp) <= 0)
	{
		result.value = kernelgen_error_compilation_failed;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot execute llc\n");
		return result;
	}
	free(command);
	
	// Write kernel IR to input stream.
	write(infp, l->kernel_source, l->kernel_source_size);
	close(infp);

	// Wait for child process to finish.
	int status;
	if (wait(&status) == -1)
	{
		result.value = kernelgen_error_compilation_failed;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Cannot synchronize with child process\n");
		return result;
	}

	// Read llc stderr stream.
	if (multiplex(errfp) > 0)
	{
		int length = 0, capacity = 0;
		char* error = NULL;
		do
		{
			capacity += SZBUF;
			error = realloc(error, capacity);
			int szread = read(errfp, error + capacity - SZBUF, SZBUF);
			length += szread;
		}
		while (length == capacity);
		close(errfp);
		
		// If error stream is not empty, take it as compilation
		// has failed.
		result.value = kernelgen_error_compilation_failed;
		kernelgen_print_error(kernelgen_launch_verbose,
			"Kernel %s compilation failed:\n%s\n",
			l->kernel_name, error);
		free(error);
		return result;
	}

	// Read llc stdout stream.
	if (multiplex(outfp) > 0)
	{
		int capacity = 0;
		*target_source = NULL;
		do
		{
			capacity += SZBUF;
			*target_source = realloc(*target_source, capacity);
			int szread = read(outfp, *target_source + capacity - SZBUF, SZBUF);
			*target_source_size += szread;
		}
		while (*target_source_size == capacity);
		close(outfp);
	}
	
	return result;
}

