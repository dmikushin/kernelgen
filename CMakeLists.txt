cmake_minimum_required (VERSION 2.6)
project (kernelgen)

set (stenfw_VERSION_MAJOR 0)
set (stenfw_VERSION_MINOR 2)

INCLUDE(CheckIncludeFile)
INCLUDE(CheckLibraryExists)
INCLUDE(CheckCSourceRuns)

set(OPT 0 CACHE STRING "Optimization level (0-3)")
set(LLVM_ROOT "$ENV{HOME}/rpmbuild/BUILD/llvm" CACHE STRING "The LLVM source+build root")
set(LLVM_MODE "Debug+Asserts" CACHE STRING "The LLVM build mode")
set(LLVM_VERSION 3.2 CACHE STRING "The LLVM version")
set(LLVM_MACROS "_GNU_SOURCE;__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS" CACHE STRING "The LLVM macros")
set(LLVM_INCLUDE "${LLVM_ROOT}/build/include" CACHE STRING "The LLVM include root")
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

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC -O${OPT}")

aux_source_directory(src/bindings bindings_sources)
add_library(bindings STATIC ${bindings_sources})

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

file(GLOB_RECURSE dyloader_sources "src/dyloader/*.cpp")
add_library(dyloader SHARED ${dyloader_sources})

aux_source_directory(src/runtime runtime_sources)
add_library(runtime SHARED ${runtime_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET runtime APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(runtime bindings bugpoint)

#frontend
