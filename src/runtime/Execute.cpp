//===- Execute.cpp - External command execution (deprecated) --------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Util.h"

#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define READ 0
#define WRITE 1
#define ERROR 2

#define SZBUF 1024

using namespace std;

static pid_t run(const char *command, int *infp, int *outfp, int *errfp) {
  int p_stdin[2], p_stdout[2], p_stderr[2];
  pid_t pid;

  if (pipe(p_stdin)) {
    fprintf(stderr, "Cannot setup pipe for stdin\n");
    return 1;
  }
  if (pipe(p_stdout)) {
    fprintf(stderr, "Cannot setup pipe for stdout\n");
    return 1;
  }
  if (pipe(p_stderr)) {
    fprintf(stderr, "Cannot setup pipe for stderr\n");
    return 1;
  }

  pid = fork();
  if (pid < 0)
    return pid;
  else if (pid == 0) {
    if (infp) {
      close(p_stdin[WRITE]);
      dup2(p_stdin[READ], READ);
    }
    if (outfp) {
      close(p_stdout[READ]);
      dup2(p_stdout[WRITE], WRITE);
    }
    if (errfp) {
      close(p_stderr[READ]);
      dup2(p_stderr[WRITE], ERROR);
    }
    if (execl("/bin/sh", "sh", "-c", command, NULL) == -1)
      exit(-1);
    exit(0);
  }

  if (infp)
    *infp = p_stdin[WRITE];
  if (outfp)
    *outfp = p_stdout[READ];
  if (errfp)
    *errfp = p_stderr[READ];

  return pid;
}

static int multiplex(int fd) {
  fd_set fdset;
  FD_ZERO(&fdset);
  FD_SET(fd, &fdset);

  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 0;

  int retval = select(fd + 1, &fdset, NULL, NULL, &tv);
  if (retval == -1) {
    fprintf(stderr, "Error while multiplexing fd = %d\n", fd);
    return -1;
  }
  return retval;
}

// Execute the specified command in the system shell, supplying
// input stream content and returning results from output and
// error streams.
int execute(string command, list<string> args, string in, string *out,
            string *err) {
  string cmd = command;
  for (list<string>::iterator it = args.begin(); it != args.end(); it++)
    cmd += " " + *it;

  int infp, outfp, errfp;
  if (run(cmd.c_str(), in.size() ? &infp : NULL, out ? &outfp : NULL,
          err ? &errfp : NULL) <=
      0) {
    cerr << "Cannot run command " << command << endl;
    return 1;
  }

  // Write to input stream.
  if (in.size()) {
    write(infp, in.c_str(), in.size());
    close(infp);
  }

  // Wait for child process to finish.
  int status;
  if (wait(&status) == -1) {
    cerr << "Cannot synchronize with child process" << endl;
    return 1;
  }
  if (status)
    return status;

  // Read error stream.
  if (err && (multiplex(errfp) > 0)) {
    int length = 0, capacity = 0;
    char *cerr = NULL;
    do {
      capacity += SZBUF;
      cerr = (char *)realloc((void *)cerr, capacity);
      int szread = read(errfp, cerr + capacity - SZBUF, SZBUF);
      length += szread;
    } while (length == capacity);
    close(errfp);
    err->assign(cerr, length);
    if (cerr)
      free(cerr);
  }

  // Read output stream.
  if (out && (multiplex(outfp) > 0)) {
    int length = 0, capacity = 0;
    char *cout = NULL;
    do {
      capacity += SZBUF;
      cout = (char *)realloc((void *)cout, capacity);
      int szread = read(outfp, cout + capacity - SZBUF, SZBUF);
      length += szread;
    } while (length == capacity);
    close(outfp);
    out->assign(cout, length);
    if (cout)
      free(cout);
  }

  return 0;
}
