/* Compiler arithmetic
   Copyright (C) 2000-2004 Free Software Foundation, Inc.
   Contributed by Andy Vaught

This file is part of G95.

G95 is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

G95 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with G95; see the file COPYING.  If not, write to
the Free Software Foundation, 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.  */

#include "g95.h"


/* The g95_integer_kinds[] and g95_real_kinds structures have
 * everything the front end needs to know about integers and real
 * numbers on the target.  Other entries of the structure are
 * calculated from these values.  The first entry is the default kind,
 * the second entry of the real structure is the default double
 * kind. */

g95_integer_info g95_integer_kinds[32] = {
  { 4,  2,  31,  32 },
  { 8,  2,  63,  64 },
  { 16, 2, 127, 128 },
  { 2,  2,  15,  16 },
  { 1,  2,   7,   8 },
  { 0,  0,   0,   0 }
};

g95_logical_info g95_logical_kinds[32] = {
  { 4,  32 },
  { 8,  64 },
  { 2,  16 },
  { 1,   8 },
  { 0,   0 }
};



g95_ff g95_real_kinds[32] = {
  { 4, 2, END_LITTLE, 32, 0, 1, 8, 127, 255, 9, 23, MSB_IMPLICIT },
  { 8, 2, END_LITTLE, 64, 0, 1, 11, 1023, 2047, 12, 52, MSB_IMPLICIT },
  { 10, 2, END_LITTLE, 80, 0, 1, 15, 16383, 32767, 16, 64, MSB_EXPLICIT },
  { 16, 2, END_LITTLE, 128, 0, 1, 15, 16383, 32767, 16, 112, MSB_IMPLICIT },
  { 0 }
};

