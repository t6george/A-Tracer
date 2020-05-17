SRC_DIR ?= src
HEADER_DIR ?= include

SOURCES := $(shell find src -name "*.cpp" -or -name "*.cc")
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))
INCLUDES := $(shell find include -type d | sed s/^/-I/)

CPPC := clang++
CPPFLAGS := -std=c++17 -g -Wall -Werror $(INCLUDES)

tracer: $(OBJECTS)
	$(CPPC) -o $@ $^ $(CPPFLAGS)

%.o: %.cpp
	$(CPPC)  $< -o $@ $(CPPFLAGS) -c

.PHONY: clean

clean:
	find . -type f -name '*.o' -delete
	rm tracer