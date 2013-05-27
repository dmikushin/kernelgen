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

#include "../DataTypes.h"
#include "../GlobalVariables.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandMemory.h"

void memorySetMaxRegister(int reg) {
  if (reg == 63)
    return;

#ifdef DebugMode
  if (reg > 63 || reg < 0)
    throw;
#endif
  if (csCurrentInstruction.OpcodeWord1 & 0x04000000) {
    reg++;
    if (reg >= 63)
      throw 148; //reg too large for .E
  }
  if (reg >= csRegCount)
    csRegCount = reg + 1;
}
inline void memorySetMaxRegisterWithoutE(int reg) {
  if (reg != 63 && reg >= csRegCount)
    csRegCount = reg + 1;
}

//Global Memory Operand
struct OperandRuleGlobalMemoryWithImmediate32 : OperandRule {
  OperandRuleGlobalMemoryWithImmediate32()
      : OperandRule(GlobalMemoryWithImmediate32) {}
  virtual void Process(SubString &component) {
    unsigned int memory;
    int register1;
    component.ToGlobalMemory(register1, memory);
    //Check max reg when register is not RZ(63)
    memorySetMaxRegister(register1);
    csCurrentInstruction.OpcodeWord0 |= register1 << 20; //RE1
    WriteToImmediate32(memory);
  }
} OPRGlobalMemoryWithImmediate32;

struct OperandRuleGlobalMemoryWithImmediate24 : OperandRule {
  OperandRuleGlobalMemoryWithImmediate24()
      : OperandRule(GlobalMemoryWithImmediate32) {}
  virtual void Process(SubString &component) {
    unsigned int memory;
    int register1;
    component.ToGlobalMemory(register1, memory);
    memorySetMaxRegisterWithoutE(register1);
    bool negative = false;
    if (memory & 0x80000000) {
      negative = true;
      memory--;
      memory ^= 0xFFFFFFFF;
    }
    if (memory > 0x7FFFFF)
      throw 150; //24-bit.
    if (negative) {
      memory ^= 0xFFFFFF;
      memory++;
      memory &= 0xFFFFFF;
    }
    csCurrentInstruction.OpcodeWord0 |= register1 << 20; //RE1
    WriteToImmediate32(memory);
  }
} OPRGlobalMemoryWithImmediate24;

//Constant Memory Operand
struct OperandRuleConstantMemory : OperandRule {
  OperandRuleConstantMemory() : OperandRule(ConstantMemory) {}
  virtual void Process(SubString &component) {
    unsigned int bank, memory;
    int register1;
    component.ToConstantMemory(bank, register1, memory, 0x1f);
    memorySetMaxRegisterWithoutE(register1);
    csCurrentInstruction.OpcodeWord0 |= register1 << 20; //RE1
    csCurrentInstruction.OpcodeWord1 |= bank << 10;
    WriteToImmediate32(memory);
    //no need to do the marking for constant memory
  }
} OPRConstantMemory;

struct OperandRuleGlobalMemoryWithLastWithoutLast2Bits : OperandRule {
  OperandRuleGlobalMemoryWithLastWithoutLast2Bits()
      : OperandRule(GlobalMemoryWithImmediate32) {}
  virtual void Process(SubString &component) {
    unsigned int memory;
    int reg1;
    component.ToGlobalMemory(reg1, memory);
    if (memory % 4 != 0)
      throw 138; //address must be multiple of 4

    memorySetMaxRegister(reg1);
    csCurrentInstruction.OpcodeWord0 |= reg1 << 20; //RE1
    WriteToImmediate32(memory);

  }
} OPRGlobalMemoryWithLastWithoutLast2Bits;

struct OperandRuleMemoryForATOM : OperandRule {
  OperandRuleMemoryForATOM() : OperandRule(Custom) {}
  virtual void Process(SubString &component) {
    unsigned int memory;
    int reg1;
    bool negative = false;
    component.ToGlobalMemory(reg1, memory);
    if (memory & 0x80000000) {
      memory--;
      memory ^= 0xFFFFFFFF;
      negative = true;
    }
    if (memory > 0x7ffff)
      throw 141; //20-bit signed integer
    if (negative) {
      memory ^= 0x000FFFFF;
      memory++;
      memory &= 0x000FFFFF;
    }
    csCurrentInstruction.OpcodeWord0 |= memory << 26;
    csCurrentInstruction.OpcodeWord1 |= (memory & 0x0001ffff) >> 6;
    csCurrentInstruction.OpcodeWord1 |= (memory & 0x000e0000) << 6;
    csCurrentInstruction.OpcodeWord0 |= reg1 << 20;
  }
} OPRMemoryForATOM;
