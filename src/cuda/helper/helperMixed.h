//===- helperMixed.h - AsFermi mixed functionality ------------------------===//
//
//     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
//        compiler for NVIDIA GPUs, targeting numerical modeling code.
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains various helper functions used by the assembler (during
// the preprocessing stage)
//
// 1: Main helper functions
// 2: Command line stage helper functions
// 9: Debugging helper functions
//
// all functions are prefixed with 'hp'
//
//===----------------------------------------------------------------------===//

#ifndef helperMixedDefined
#define helperMixedDefined

#include <fstream>

using namespace std;

//	1
//-----Main helper functions
void hpCleanUp();
//-----End of main helper functions

//	2
//----- Command-line stage helper functions
void hpUsage();
int hpHexCharToInt(char* str);
int hpFileSizeAndSetBegin(fstream &file);
int hpFindInSource(char target, int startPos, int &length);
void hpReadSource(char* path);
void hpReadSourceArray(char* source);
void hpCheckOutputForReplace(char* path, char* kernelname, char* replacepoint);
//-----End of command-line helper functions

//9
//-----Debugging functions
void hpPrintLines();
void hpPrintInstructions();
void hpPrintDirectives();	
void hpPrintComponents(Instruction &instruction);
static const char* binaryRef[16] = {"0000", "1000", "0100", "1100", "0010", "1010", "0110", "1110", 
									"0001", "1001", "0101", "1101", "0011", "1011", "0111", "1111"};
void hpPrintBinary8(unsigned int word0, unsigned int word1);
//-----End of debugging functions

//Convert binary string often seen on asfermi's site into an unsigned int
void hpBinaryStringToOpcode4(char* string, unsigned int &word0, int &i);
void hpBinaryStringToOpcode4(char* string, unsigned int &word0);
void hpBinaryStringToOpcode8(char* string, unsigned int &word0, unsigned int &word1);

#else
#endif
