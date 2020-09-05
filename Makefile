SRC_DIR ?= src
HEADER_DIR ?= include
CPPC := 
CPPFLAGS := 
NOINCFLAG := 
LINKFLAG :=

SOURCES := $(shell find src -name "*.cu")
INCLUDES := $(shell find include -type d | sed s/^/-I/)
CPPC := nvcc
CPPFLAGS := -g $(INCLUDES) --expt-relaxed-constexpr

ifeq ($(target), gpu)
	CPPFLAGS += -D GPU=1
else
	CPPFLAGS += -D GPU=0
endif

ifeq ($(sample), montecarlo)
        CPPFLAGS += -D MONTE_CARLO=1
else
        CPPFLAGS += -D MONTE_CARLO=0
endif

OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

export CXX=$(CPPC)

tracer: $(OBJECTS)
	$(CPPC) -o $@ $^ $(CPPFLAGS)

%.o: %.cu
	$(CPPC) $(NOINCFLAG) $< -o $@ $(CPPFLAGS) -dc

.PHONY: clean
clean:
	find . -type f -name '*.o' -delete
	@rm tracer 2>/dev/null || true
