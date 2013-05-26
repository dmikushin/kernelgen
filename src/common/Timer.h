//===- Timer.h - KernelGen time measurement API ---------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements time measurement API.
//
//===----------------------------------------------------------------------===//

#include <time.h>

#include "llvm/Support/Timer.h"
#include <map>

namespace kernelgen {
namespace utils {

class timer {
  timespec time_start, time_stop;
  bool started;

public:

  static timespec get_resolution();

  timer(bool start = true);

  timespec start();
  timespec stop();

  double get_elapsed(timespec *start = NULL);
};

class TimingInfo {
  std::map<llvm::StringRef, llvm::Timer *> TimingData;
  llvm::TimerGroup TG;

public:
  TimingInfo(llvm::StringRef GroupName) : TG(GroupName) {}

  // TimingDtor - Print out information about timing information
  ~TimingInfo() {
    // Delete all of the timers, which accumulate their info into the
    // TimerGroup.
    for (std::map<llvm::StringRef, llvm::Timer *>::iterator
             I = TimingData.begin(),
             E = TimingData.end();
         I != E; ++I)
      delete I->second;
    // TimerGroup is deleted next, printing the report.
  }

  /// getTimer - Return the timer with the specified name, if verbose mode is
  /// set
  /// to make perf reports.
  llvm::Timer *getTimer(llvm::StringRef TimerName);
};

}
}
