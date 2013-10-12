/*
 * Copyright (c) 2012 by Dmitry Mikushin
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

//
// Loader source code template.
// Note the loader code regcount must be set in UBERKERN_LOADER_REGCOUNT.
// Kernel prototype: 
//     uberkern(void* user_pointer, uint32_t szbinary, char* binary,
//         int32_t command, int32_t address);
//
// szbinary - dynamic kernel binary size
// binary - dynamic kernel binary to load
//          (if not yet loaded).
// command - select command:
//           0 - load effective PC of uberkern
//           1 - load dynamic kernel source code
//           other value - execute dynamic kernel
//           (to be used as JMP address in entry points)
// address - the dynamic kernel relative offset in uberkern
//
static const char *uberkern[] = {
  "!Machine 64",
  "!Kernel uberkern",
  "!Param 256 1",
  "LEPC R0",                             // R0 = LEPC

  "MOV R2, c[0x0][$COMMAND]",            // Check if the "command" contains 0.
  "ISETP.NE.AND P0, pt, R2, RZ, pt",
  "@P0 BRA !LD",                         // If not - go to !LD
  "MOV R2, c[0x0][$BINARYLO]",
  "MOV R3, c[0x0][$BINARYHI]",
  "ST.E [R2], R0",                       // If yes, write LEPC to "binary" and exit.
  "EXIT",
  "!Label LD",
  "MOV R1, 0x1",                         // Check if the "command" contains 1.
  "ISETP.NE.AND P0, pt, R2, R1, pt",
  "@P0 BRA !BRA",                        // If not - go to !BRA
                                         // If yes, write dynamic kernel code and exit.
                                         // Load the dynamic kernel starting address.
  "!Label GO",
  
  "MOV R1, c[0x0][$ADDRESS]",            // R1 = ADDRESS        <-- 4-byte value of goto offset
  "IADD R1, R1, 0x130",                  // R1 += !FRE
  "IADD R0, R0, R1",                     // R0 += R1
  "MOV R1, 0x1",                         // R1 = 1                  <-- low word compound = 1

  // Load kernel's size and then load each instruction in a loop.
  // TODO do this in multi-thread kernel

  "MOV R4, c[0x0][$BINARYLO]",
  "MOV R5, c[0x0][$BINARYHI]",           // (R2, R3) := binary
  "MOV R6, c[0x0][$SZBINARY]",           // R6 := szbinary
  "!Label L1",
  "ISETP.EQ.AND P0, pt, R6, RZ, pt",     // if (R6 == 0)
  "@P0 EXIT",                            // exit;
                                         // else
                                         // {
                                         //     Load instructions from args to kernel space                                         
  "LD.E.64 R2, [R4]",                    //     *(R2, R3) = (R4, R5)
  "ST.E.64 [R0], R2",                    //     *(R0, R1) = (R2, R3)
  "IADD R0, R0, 8",                      //     R0 += 8
  "IADD R4, R4, 8",                      //     R4 += 8
  "IADD R6, R6, -8",                     //     R6 -= 8
  "BRA !L1",                             //     goto !L1
                                         // }
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "NOP",
  "!Label BRA",
  
  "BRA c[0x0][$ADDRESS]",                // goto dynamic kernel "address" offset
  "!Label FRE",
  "$BUF",                                // $BUF more NOPs here as free space for code insertions
  "!EndKernel"
};

