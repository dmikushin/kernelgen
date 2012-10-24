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

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/bugpoint")
include_directories("${PROJECT_SOURCE_DIR}/src/frontend")
include_directories("${PROJECT_SOURCE_DIR}/src/gpu")
include_directories("${PROJECT_SOURCE_DIR}/src/include")
include_directories("${PROJECT_SOURCE_DIR}/src/runtime")

include_directories("${LLVM_INCLUDE}")
include_directories("${BUGPOINT_ROOT}")
include_directories("${POLLY_ROOT}/include")
include_directories("${POLLY_INCLUDE}")
include_directories("${CLOOG_INCLUDE}")
include_directories("${ISL_INCLUDE}")
include_directories("/usr/include/i686-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu")
include_directories("/usr/lib64/libffi-3.0.9/include")
include_directories("/usr/lib64/libffi-3.0.10/include")

link_directories("${LLVM_ROOT}/build/${LLVM_MODE}/lib")
link_directories("${POLLY_ROOT}/${LLVM_MODE}/lib")

set(CMAKE_INSTALL_RPATH "/opt/kernelgen/lib")

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
# GPU BINDINGS
#
file(GLOB_RECURSE gpu_sources "src/gpu/*.cpp")
add_library(kernelgen-gpu SHARED ${gpu_sources})

#
# RUNTIME LIBRARY
#
aux_source_directory(src/runtime runtime_sources)
add_library(kernelgen-rt SHARED ${runtime_sources})
set_property(TARGET kernelgen-rt APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-rt APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-rt bugpoint kernelgen-gpu LLVM-${LLVM_VERSION}svn LLVMPolly)

#frontend

#
# VERSION PRINTER
#
aux_source_directory(src/version version_sources)
add_executable(kernelgen-version ${version_sources})
set_property(TARGET kernelgen-version APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")

