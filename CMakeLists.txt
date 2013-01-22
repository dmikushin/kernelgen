##===- CMakeLists.txt - KernelGen CMake build script ----------------------===//
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===//

cmake_minimum_required (VERSION 2.6)
project (kernelgen)

set(stenfw_VERSION_MAJOR 0)
set(stenfw_VERSION_MINOR 2)

#
# Do not allow CMake to strip RPATHs, because KernelGen uses it
# to find its library dir and its own version of GLIBC.
#
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

execute_process(COMMAND svnversion -n ${PROJECT_SOURCE_DIR}/src OUTPUT_VARIABLE KERNELGEN_VERSION)
message("Compiling KernelGen version ${KERNELGEN_VERSION}")

set(OPT 0 CACHE STRING "Optimization level (0-3)")
set(RPM_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/../../.." CACHE STRING "The RPM install prefix")
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
include_directories("${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/include/")
include_directories("${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/include/libelf")
include_directories("${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib/libffi-3.0.11/include")
include_directories("/usr/include/i686-linux-gnu")
include_directories("/usr/include/x86_64-linux-gnu")

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
link_directories("${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib")
link_directories("${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib64")

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-g -fPIC -O${OPT}")

#
# IMPORT STATIC LIBRARIES
#
add_library(libelf STATIC IMPORTED)
set_property(TARGET libelf PROPERTY
	IMPORTED_LOCATION "${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib/libelf.a")
add_library(libffi STATIC IMPORTED)
set_property(TARGET libffi PROPERTY
	IMPORTED_LOCATION "${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib/libffi.a")
add_library(libmhash STATIC IMPORTED)
set_property(TARGET libmhash PROPERTY
	IMPORTED_LOCATION "${RPM_INSTALL_PREFIX}/opt/kernelgen/usr/lib/libmhash.a")

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
target_link_libraries(kernelgen-simple bugpoint common frontend LLVM-${LLVM_VERSION}svn libelf iberty rt)

#
# GCC COMPILE-TIME PLUGIN
#
aux_source_directory(src/plugin plugin_sources)
add_library(kernelgen-ct SHARED ${plugin_sources})
set_property(TARGET kernelgen-ct APPEND PROPERTY COMPILE_DEFINITIONS "KERNELGEN_VERSION=\"${KERNELGEN_VERSION}\"")
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-ct APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-ct bugpoint frontend common LLVM-${LLVM_VERSION}svn rt)

#
# CUDA BINDINGS
#
file(GLOB_RECURSE cuda_sources "src/cuda/*.cpp")
add_library(kernelgen-cuda SHARED ${cuda_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-cuda APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-cuda common LLVM-${LLVM_VERSION}svn libelf)
list(REMOVE_ITEM cuda_sources
	"${PROJECT_SOURCE_DIR}/src/cuda/cuda_dyloader.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/Cuda.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/AlignData.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/MergeCubins.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/InsertLEPC.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/GetLoadEffectiveLayout.cpp"
	"${PROJECT_SOURCE_DIR}/src/cuda/ResolveExternalCalls.cpp")
add_executable(kernelgen-asfermi ${cuda_sources})
foreach(DEFINE ${LLVM_MACROS})
	set_property(TARGET kernelgen-asfermi APPEND PROPERTY COMPILE_DEFINITIONS "${DEFINE}")
endforeach()
target_link_libraries(kernelgen-asfermi)

#
# CUDA LLVM IR -LEVEL INCLUDES
#
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/monitor.bc COMMAND ${CULANG} ${CULANG_INCLUDE_ARGS}
	${PROJECT_SOURCE_DIR}/src/cuda/include/monitor.cu -o - | opt -O3 -o ${PROJECT_BINARY_DIR}/monitor.bc)
add_custom_target(cuda_monitor ALL DEPENDS ${PROJECT_BINARY_DIR}/monitor.bc)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/runtime.bc COMMAND ${CULANG} ${CULANG_INCLUDE_ARGS}
	${PROJECT_SOURCE_DIR}/src/cuda/include/runtime.cu -o - | opt -O3 -o ${PROJECT_BINARY_DIR}/runtime.bc)
add_custom_target(cuda_runtime ALL DEPENDS ${PROJECT_BINARY_DIR}/runtime.bc)
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/math.bc COMMAND opt
	${PROJECT_SOURCE_DIR}/src/cuda/include/math.bc -o ${PROJECT_BINARY_DIR}/math.bc)
add_custom_target(cuda_math ALL DEPENDS ${PROJECT_BINARY_DIR}/math.bc)

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
target_link_libraries(kernelgen-rt bugpoint common kernelgen-cuda LLVMPolly LLVM-${LLVM_VERSION}svn libelf rt gmp libmhash dl libffi)

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
# BUILD WRAPPER (SHOULD BE USED WHEN BUILDING PARTS
# OF KERNELGEN IN BOOTSTRAPPED MODE).
#
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/kernelgen-build
	COMMAND cp ${PROJECT_SOURCE_DIR}/src/build/kernelgen-build ${PROJECT_BINARY_DIR}/kernelgen-build)
add_custom_target(kernelgen-build ALL DEPENDS ${PROJECT_BINARY_DIR}/kernelgen-build)

#
# KERNELGEN BUILDBOT (BURAN)
#
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/kernelgen-buran
	COMMAND cp ${PROJECT_SOURCE_DIR}/src/buran/kernelgen-buran ${PROJECT_BINARY_DIR}/kernelgen-buran)
add_custom_target(kernelgen-buran ALL DEPENDS ${PROJECT_BINARY_DIR}/kernelgen-buran)

#
# INSTALL
#
# bin
INSTALL(TARGETS
	kernelgen-asfermi kernelgen-simple kernelgen-version
	DESTINATION bin)
INSTALL(PROGRAMS
	${PROJECT_BINARY_DIR}/kernelgen-build
	${PROJECT_BINARY_DIR}/kernelgen-buran
	${PROJECT_BINARY_DIR}/kernelgen-dragonegg
	${PROJECT_BINARY_DIR}/kernelgen-vars.sh
	DESTINATION bin)
# lib
INSTALL(TARGETS
	kernelgen-ct kernelgen-rt kernelgen-cuda
	DESTINATION lib)
# include/cuda
INSTALL(FILES
	${PROJECT_BINARY_DIR}/math.bc
	${PROJECT_BINARY_DIR}/monitor.bc
	${PROJECT_BINARY_DIR}/runtime.bc
	DESTINATION include/cuda)
