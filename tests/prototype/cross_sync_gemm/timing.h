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

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

#pragma pack(push, 1)

// The built-in timer value type.
typedef struct
{
	int64_t seconds;
	int64_t nanoseconds;
}
util_time_t;

#pragma pack(pop)

// Get the built-in timer resolution.
void util_get_timer_resolution(util_time_t* val);

// Get the built-in timer value.
void util_get_time(util_time_t* val);

// Get the built-in timer measured values difference.
double util_get_time_diff(util_time_t* val1, util_time_t* val2);

// Print the built-in timer measured values difference.
void util_print_time_diff(util_time_t* val1, util_time_t* val2);

#ifdef __cplusplus
}
#endif

#endif // TIME_H

