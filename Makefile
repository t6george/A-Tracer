SRC_DIR ?= src
HEADER_DIR ?= include
target ?= cpu
INCLUDES := $(shell find include -type d | sed s/^/-I/)
ifeq($(target), gpu)
	CPPC := nvcc
    CPPFLAGS := -g $(INCLUDES)
    SOURCES := $(shell find src -name "*.cpp" -or -name "*.cu")
else
    CPPC := g++
    CPPFLAGS := -g -std=c++17 -g -Wall -Werror $(INCLUDES)
    SOURCES := $(shell find src -name "*.cpp")
endif
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))
tracer: $(OBJECTS)
    $(CPPC) -o $@ $^ $(CPPFLAGS)
%.o: %.cpp
    $(CPPC)  $< -o $@ $(CPPFLAGS) -c
%.o: %.cu
    $(CPPC)  $< -o $@ $(CPPFLAGS) -c
.PHONY: clean
clean:
    find . -type f -name '*.o' -delete
    rm tracer