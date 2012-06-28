LLVM_ROOT ?= ${HOME}/rpmbuild/BUILD/llvm
LLVM_MODE ?= Debug+Asserts
LLVM_VERSION ?= 3.2
LLVM_DEF = -D_GNU_SOURCE -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS
LLVM_INC = -I$(LLVM_ROOT)/build/include $(LLVM_DEF)

BUGPOINT_ROOT ?= $(LLVM_ROOT)/tools/bugpoint

BUGPOINT := BugDriver CrashDebugger ExtractFunction Miscompilation ToolRunner \
	bugpoint ExecutionDriver FindBugs OptimizerDriver
BUGPOINT := $(addsuffix .bugpoint, $(BUGPOINT))

OBJECTS := $(addsuffix .o, $(BUGPOINT))

all: libbugpoint.a

%.bugpoint.o: $(BUGPOINT_ROOT)/%.cpp
	$(CXX) -Dmain=bugpoint $(CXXFLAGS) $(LLVM_INC) -c $< -o $@

libbugpoint.a: $(OBJECTS)
	$(AR) rcs $@ $?

clean:
	rm -rf libbugpoint.a *.o

