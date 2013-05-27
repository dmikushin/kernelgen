/*
 * Copyright (c) 2011, 2012 by Hou Yunqing
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "GlobalVariables.h"

#include "stdafx.h" //Mark

bool csSelfDebug = false;

int csLineNumber = 0;
int csInstructionOffset;
Line csCurrentLine;
Instruction csCurrentInstruction;
Directive csCurrentDirective;

string ofilename;
char *csSource;
int csSourceSize;
int csRegCount = 0;
int csBarCount = 0;
bool csAbsoluteAddressing = true;

OperationMode csOperationMode = Undefined;
char *csSourceFilePath;

bool csExceptionPrintUsage = false;
bool csErrorPresent = false;

//the following 3 variables are for Replace mode.
int csOutputSectionOffset;
int csOutputSectionSize;
int csOutputInstructionOffset;

list<MasterParser *>
    csMasterParserList; //List of parsers to loaded at initialization
list<LineParser *> csLineParserList;
list<InstructionParser *> csInstructionParserList;
list<DirectiveParser *> csDirectiveParserList;

InstructionRule **csInstructionRules; //sorted array
int *csInstructionRuleIndices; //Instruction name index of the corresponding
                               //element in csInstructionRules
int csInstructionRuleCount;
list<InstructionRule *> csInstructionRulePrepList; //used for preperation

DirectiveRule **csDirectiveRules; //sorted array
int *csDirectiveRuleIndices; //Directive name index of the corresponding element
                             //in csDirectiveRules
int csDirectiveRuleCount;
list<DirectiveRule *> csDirectiveRulePrepList; //used for preperation

stack<MasterParser *> csMasterParserStack;
stack<LineParser *> csLineParserStack;
stack<InstructionParser *> csInstructionParserStack;
stack<DirectiveParser *> csDirectiveParserStack;

MasterParser *csMasterParser; //curent Master Parser
LineParser *csLineParser;
InstructionParser *csInstructionParser;
DirectiveParser *csDirectiveParser;

list<Line> csLines;
list<Instruction> csInstructions;
list<Directive> csDirectives;
list<Label> csLabels;
list<LabelRequest> csLabelRequests;

//=================

ELFSection cubinSectionEmpty, cubinSectionSHStrTab, cubinSectionStrTab,
    cubinSectionSymTab;
ELFSection cubinSectionConstant2, cubinSectionNVInfo;
ELFSegmentHeader cubinSegmentHeaderPHTSelf;
ELFSegmentHeader cubinSegmentHeaderConstant2;

unsigned int cubinCurrentSectionIndex = 0;
unsigned int cubinCurrentOffsetFromFirst =
    0; //from the end of the end of .symtab
unsigned int cubinCurrentSHStrTabOffset = 0;
unsigned int cubinCurrentStrTabOffset = 0;
unsigned int cubinTotalSectionCount = 0;
unsigned int cubinPHTOffset = 0;
unsigned int cubinPHCount;
unsigned int cubinConstant2Size = 0;
unsigned int cubinCurrentConstant2Offset = 0;
bool cubinConstant2Overflown = false;

void (*cubinCurrentConstant2Parser)(SubString &content);

Architecture cubinArchitecture = sm_20; //default architecture is sm_20
bool cubin64Bit = false;

char *cubin_str_empty = "";
char *cubin_str_shstrtab = ".shstrtab";
char *cubin_str_strtab = ".strtab";
char *cubin_str_symtab = ".symtab";
char *cubin_str_extra1 = ".nv.global.init";
char *cubin_str_extra2 = ".nv.global";
char *cubin_str_text = ".text.";
char *cubin_str_constant0 = ".nv.constant0.";
char *cubin_str_info = ".nv.info.";
char *cubin_str_shared = ".nv.shared.";
char *cubin_str_local = ".nv.local.";
char *cubin_str_constant = ".nv.constant";
char *cubin_str_nvinfo = ".nv.info";

bool csCurrentKernelOpened = false;
Kernel csCurrentKernel;
list<Kernel> csKernelList;
list<Constant2> csConstant2List;
