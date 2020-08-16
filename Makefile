SRC_DIR ?= src
HEADER_DIR ?= include
CPPC := 
CPPFLAGS := 
NOINCFLAG := 

SOURCES := $(shell find src -name "*.cpp")
INCLUDES := $(shell find include -type d | sed s/^/-I/)

ifeq ($(target), gpu)
	CPPC := nvcc
	CPPFLAGS := -g $(INCLUDES) -D GPU=1
	SOURCES += $(shell find src -name "*.cu")
else
	CPPC := clang++
	CPPFLAGS := -g -std=c++17 -Wall -Werror $(INCLUDES) -D GPU=0
	NOINCFLAG := -nocudainc -nocudalib
endif

OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

ifeq ($(sample), montecarlo)
	CPPFLAGS += -D MONTE_CARLO=1
else
	CPPFLAGS += -D MONTE_CARLO=0
endif

export CXX=$(CPPC)

tracer: $(OBJECTS)
	$(CPPC) -o $@ $^ $(CPPFLAGS)

%.o: %.cpp
	$(CPPC) $< -o $@ $(CPPFLAGS) -c

%.o: %.cu
	$(CPPC) $(NOINCFLAG) $< -o $@ $(CPPFLAGS) -c

.PHONY: clean
clean:
	find . -type f -name '*.o' -delete
	@rm tracer 2>/dev/null || true
