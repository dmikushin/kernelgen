#
# KGen - the LLVM-based compiler with GPU kernels generation over C backend.
#
# Copyright (c) 2011 Dmitry Mikushin
#
# This software is provided 'as-is', without any express or implied warranty.
# In no event will the authors be held liable for any damages arising 
# from the use of this software.
# Permission is granted to anyone to use this software for any purpose, 
# including commercial applications, and to alter it and redistribute it freely,
# subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented;
# you must not claim that you wrote the original software.
# If you use this software in a product, an acknowledgment
# in the product documentation would be appreciated but is not required.
# 2. Altered source versions must be plainly marked as such,
# and must not be misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#

cmake_minimum_required (VERSION 2.6)
project (kernelgen)

set (stenfw_VERSION_MAJOR 0)
set (stenfw_VERSION_MINOR 2)

#INCLUDE(CheckIncludeFile)
#INCLUDE(CheckLibraryExists)
#INCLUDE(CheckCSourceRuns)

execute_process(COMMAND svnversion -n ${PROJECT_SOURCE_DIR}/src OUTPUT_VARIABLE KERNELGEN_VERSION)
message("Compiling KernelGen version ${KERNELGEN_VERSION}")

set(OPT 0 CACHE STRING "Optimization level (0-3)")
set(BOOTSTRAP_ROOT "${CMAKE_INSTALL_PREFIX}" CACHE STRING "The KernelGen bootsrap root")
set(RPM_ROOT "$ENV{HOME}/rpmbuild/BUILD" CACHE STRING "The RPM build root")
set(LLVM_ROOT "${RPM_ROOT}/llvm" CACHE STRING "The LLVM source+build root")
set(LLVM_MODE "Debug+Asserts" CACHE STRING "The LLVM build mode")
set(LLVM_VERSION 3.2 CACHE STRING "The LLVM version")
set(LLVM_MACROS "_GNU_SOURCE;__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS" CACHE STRING "The LLVM macros")
set(LLVM_INCLUDE "${LLVM_ROOT}/build/include" CACHE STRING "The LLVM include root")
set(POLLY_ROOT "${LLVM_ROOT}/build/tools/polly" CACHE STRING "The Polly build root")
set(POLLY_INCLUDE "${LLVM_ROOT}/tools/polly/include" CACHE STRING "The LLVM/Polly include root")
set(CLOOG_INCLUDE "${RPM_ROOT}/cloog/include" CACHE STRING "The CLooG include root")
set(ISL_INCLUDE "${RPM_ROOT}/cloog/isl/include" CACHE STRING "The ISL include root")
set(BUGPOINT_ROOT "${LLVM_ROOT}/tools/bugpoint" CACHE STRING "The LLVM/bugpoint source root")
set(GCC_ROOT "${RPM_ROOT}/gcc-snapshot" CACHE STRING "The GCC source root")
set(CULANG clang -cc1 -emit-llvm -fcuda-is-device -triple nvptx64-unknown-unknown CACHE STRING
	"Clang command to compile CUDA-specific LLVM IR modules")

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/bugpoint")
include_directories("${PROJECT_SOURCE_DIR}/src/common")
include_directories("${PROJECT_SOURCE_DIR}/src/cuda")
include_directories("${PROJECT_SOURCE_DIR}/src/frontend")
include_directories("${PROJECT_SOURCE_DIR}/src/include")
include_directories("${PROJECT_SOURCE_DIR}/src/runtime")
include_directories("${GCC_ROOT}/gcc/")
include_directories("${GCC_ROOT}/build/gcc/")
include_directories("${GCC_ROOT}/include/")
include_directories("${GCC_ROOT}/libcpp/include/")

include_directories("${LLVM_INCLUDE}")
include_directories("${BUGPOINT_ROOT}")
include_directories("${POLLY_ROOT}/include")
include_directories("${POLLY_INCLUDE}")
include_directories("${CLOOG_INCLUDE}")
include_directories("${ISL_INCLUDE}")
include_directories("${CMAKE_INSTALL_PREFIX}/include/libelf")
include_directories("/usr/include/i686-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu")
include_directories("/usr/lib64/libffi-3.0.9/include")
include_directories("/usr/lib64/libffi-3.0.10/include")

#
# DUPLICATE INCLUDES FOR CLANG CUSTOM RULE
#
set(CULANG_INCLUDE_ARGS)
get_property(INCLUDES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${INCLUDES})
	list(APPEND CULANG_INCLUDE_ARGS -I${dir})
endforeach()

link_directories("${LLVM_ROOT}/build/${LLVM_MODE}/lib")
link_directories("${POLLY_ROOT}/${LLVM_MODE}/lib")
link_directories("${BOOTSTRAP_ROOT}/lib")
link_directories("${BOOTSTRAP_ROOT}/lib64")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC -O${OPT}")

#
# BUGPOINT AUTO-TESTING TOOL
#
aux_source_directory(src/bugpoint bugpoint_sources)
add_library(bugpoint STATIC ${bugpoint_sources}
	${BUGPOINT_ROOT}/BugDriver.cpp
	${BUGPOINT_ROOT}/CrashDebugger.cpp
	${BUGPOINT_ROOT}/ExtractFunction.cpp
	${BUGPOINT_ROOT}/Miscompilation.cpp
	${BUGPOINT_ROOT}/ToolRunner.cpp
	${BUGPOINT_ROOT}/bugpoint.cpp
	${BUGPOINT_ROOT}/ExecutionDriver.cpp
	${BUGPOINT_ROOT}/FindBugs.cpp
	${BUGPOINT_ROOT}/OptimizerDriver.cpp)
set_property(TARGET bugpoint APPEND PROPERTY COMPILE_DEFINITIONS "main=bugpoint")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET bugpoint APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()

#
# COMMON LIBRARY
#
aux_source_directory(src/common common_sources)
add_library(common STATIC ${common_sources})
set_property(SOURCE src/common/GlobalDependences.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET common APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()

#
# FRONTEND LIBRARY
#
aux_source_directory(src/frontend frontend_sources)
add_library(frontend STATIC ${frontend_sources})
set_property(SOURCE src/frontend/BranchedLoopExtractor.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET frontend APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()

#
# SIMPLE FRONTEND & LTO LINKER BACKEND
#
aux_source_directory(src/simple simple_sources)
add_executable(kernelgen-simple ${simple_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-simple APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-simple bugpoint common frontend LLVM-${LLVM_VERSION}svn elf iberty)

#
# GCC COMPILE-TIME PLUGIN
#
aux_source_directory(src/plugin plugin_sources)
add_library(kernelgen-ct SHARED ${plugin_sources})
set_property(TARGET kernelgen-ct APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-ct APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-ct bugpoint frontend LLVM-${LLVM_VERSION}svn)

#
# CUDA BINDINGS
#
file(GLOB_RECURSE cuda_sources "src/cuda/*.cpp")
add_library(kernelgen-cuda SHARED ${cuda_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-cuda APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-cuda common LLVM-${LLVM_VERSION}svn elf)
list(REMOVE_ITEM cuda_sources
	"${PROJECT_SOURCE_DIR}/src/cuda/cuda_dyloader.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/Cuda.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/AlignData.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/MergeCubins.cpp")
add_executable(kernelgen-asfermi ${cuda_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-asfermi APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-asfermi)

#
# CUDA LLVM IR -LEVEL INCLUDES
#
file(COPY ${PROJECT_SOURCE_DIR}/src/cuda/include/math.bc DESTINATION ${PROJECT_BINARY_DIR}/cuda)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/cuda/monitor.bc COMMAND ${CULANG} ${CULANG_INCLUDE_ARGS}
	${PROJECT_SOURCE_DIR}/src/cuda/include/monitor.cu -o - | opt -O3 -S -o ${PROJECT_BINARY_DIR}/cuda/monitor.bc)
add_custom_target(cuda_monitor ALL DEPENDS ${PROJECT_BINARY_DIR}/cuda/monitor.bc)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/cuda/runtime.bc COMMAND ${CULANG} ${CULANG_INCLUDE_ARGS}
	${PROJECT_SOURCE_DIR}/src/cuda/include/runtime.cu -o - | opt -O3 -S -o ${PROJECT_BINARY_DIR}/cuda/runtime.bc)
add_custom_target(cuda_runtime ALL DEPENDS ${PROJECT_BINARY_DIR}/cuda/runtime.bc)

#
# RUNTIME LIBRARY
#
aux_source_directory(src/runtime runtime_sources)
add_library(kernelgen-rt SHARED ${runtime_sources})
set_property(SOURCE src/runtime/ConstantSubstitution.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
set_property(SOURCE src/runtime/SizeOfLoops.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
set_property(SOURCE src/runtime/RuntimeAliasAnalysis.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
set_property(SOURCE src/runtime/ScopInspection.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
set_property(SOURCE src/runtime/TransformAccesses.cpp APPEND PROPERTY COMPILE_FLAGS "-fno-rtti")
set_property(TARGET kernelgen-rt APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-rt APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-rt bugpoint common kernelgen-cuda LLVMPolly LLVM-${LLVM_VERSION}svn elf rt gmp mhash dl ffi)

#
# VERSION PRINTER
#
aux_source_directory(src/version version_sources)
add_executable(kernelgen-version ${version_sources})
set_property(TARGET kernelgen-version APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")

#
# DRAGONEGG WRAPPER
#
file(COPY ${PROJECT_SOURCE_DIR}/src/dragonegg/kernelgen-dragonegg DESTINATION ${PROJECT_BINARY_DIR})

#
# ENVIRONMENT VARS SCRIPT
#
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/kernelgen-vars.sh
	COMMAND sed s!CMAKE_INSTALL_PREFIX!${CMAKE_INSTALL_PREFIX}!g ${PROJECT_SOURCE_DIR}/src/vars/kernelgen-vars.sh >${PROJECT_BINARY_DIR}/kernelgen-vars.sh)
add_custom_target(kernelgen_vars ALL DEPENDS ${PROJECT_BINARY_DIR}/kernelgen-vars.sh)

#
# INSTALL
#
# bin
INSTALL(TARGETS
	kernelgen-asfermi kernelgen-simple kernelgen-version
	DESTINATION bin)
INSTALL(PROGRAMS
	${PROJECT_BINARY_DIR}/kernelgen-dragonegg
	${PROJECT_BINARY_DIR}/kernelgen-vars.sh
	DESTINATION bin)
# lib
INSTALL(TARGETS
	kernelgen-ct kernelgen-rt kernelgen-cuda
	DESTINATION lib)
# include/cuda
INSTALL(FILES
	${PROJECT_BINARY_DIR}/cuda/math.bc
	DESTINATION include/cuda)
INSTALL(FILES
	${PROJECT_BINARY_DIR}/cuda/monitor.bc
	${PROJECT_BINARY_DIR}/cuda/runtime.bc
	DESTINATION include/cuda)
