cmake_minimum_required (VERSION 2.6)
project (kernelgen)

set (stenfw_VERSION_MAJOR 0)
set (stenfw_VERSION_MINOR 2)

#INCLUDE(CheckIncludeFile)
#INCLUDE(CheckLibraryExists)
#INCLUDE(CheckCSourceRuns)

add_custom_command(OUTPUT kernelgen_version COMMAND svnversion -n)

set(OPT 0 CACHE STRING "Optimization level (0-3)")
set(LLVM_ROOT "$ENV{HOME}/rpmbuild/BUILD/llvm" CACHE STRING "The LLVM source+build root")
set(LLVM_MODE "Debug+Asserts" CACHE STRING "The LLVM build mode")
set(LLVM_VERSION 3.2 CACHE STRING "The LLVM version")
set(LLVM_MACROS "_GNU_SOURCE;__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS" CACHE STRING "The LLVM macros")
set(LLVM_INCLUDE "${LLVM_ROOT}/build/include" CACHE STRING "The LLVM include root")
set(POLLY_ROOT "$ENV{HOME}/rpmbuild/BUILD/llvm/build/tools/polly/include" CACHE STRING "The Polly build root")
set(POLLY_INCLUDE "$ENV{HOME}/rpmbuild/BUILD/llvm/tools/polly/include" CACHE STRING "The LLVM/Polly include root")
set(CLOOG_INCLUDE "$ENV{HOME}/rpmbuild/BUILD/cloog/include" CACHE STRING "The CLooG include root")
set(ISL_INCLUDE "$ENV{HOME}/rpmbuild/BUILD/cloog/isl/include" CACHE STRING "The ISL include root")
set(BUGPOINT_ROOT "${LLVM_ROOT}/tools/bugpoint" CACHE STRING "The LLVM/bugpoint source root")

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/src/bindings")
include_directories("${PROJECT_SOURCE_DIR}/src/bugpoint")
include_directories("${PROJECT_SOURCE_DIR}/src/dyloader")
include_directories("${PROJECT_SOURCE_DIR}/src/frontend")
include_directories("${PROJECT_SOURCE_DIR}/src/include")
include_directories("${PROJECT_SOURCE_DIR}/src/runtime")

include_directories("${LLVM_INCLUDE}")
include_directories("${BUGPOINT_ROOT}")
include_directories("${POLLY_ROOT}")
include_directories("${POLLY_INCLUDE}")
include_directories("${CLOOG_INCLUDE}")
include_directories("${ISL_INCLUDE}")
include_directories("/usr/include/i686-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu")
include_directories("/usr/lib64/libffi-3.0.9/include")
include_directories("/usr/lib64/libffi-3.0.10/include")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC -O${OPT}")

file(GLOB_RECURSE gpu_sources "src/gpu/*.cpp")
add_library(kernelgen-gpu STATIC ${gpu_sources})

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

aux_source_directory(src/runtime runtime_sources)
add_library(kernelgen-rt SHARED ${runtime_sources})
set_property(TARGET runtime APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=${KERNELGEN_VERSION}")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET runtime APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(runtime bindings bugpoint)

#frontend
