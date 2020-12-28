BITS = 64

ODIR  = obj64
tmp  := $(shell mkdir -p $(ODIR))

# Basic compiler configuration and flags
HOME     = /home/ziz221
TBB_INSTALL_DIR = $(HOME)/tbb2019_20190605oss_lin/tbb2019_20190605oss
TBB_INCLUDE = $(TBB_INSTALL_DIR)/include

CXX      = gcc 
CXXFLAGS = -MMD -ggdb -O3 -std=c++11 -m$(BITS) -ltbb -mavx
LDFLAGS	 = -m$(BITS) -lpthread -lrt -ltbb -mavx

# The basenames of the c++ files that this program uses
CXXFILES_PARALLEL = kmeans_parallel
CXXFILES_SERIAL = kmeans_serial

# The executable we will build
TARGET_PARALLEL = $(ODIR)/kmeans_parallel
TARGET_SERIAL = $(ODIR)/kmeans_serial

# Create the .o names from the CXXFILES
OFILES_PARALLEL = $(patsubst %, $(ODIR)/%.o, $(CXXFILES_PARALLEL))
OFILES_SERIAL = $(patsubst %, $(ODIR)/%.o, $(CXXFILES_SERIAL))

# Create .d files to store dependency information, so that we don't need to
# clean every time before running make
DFILES_PARALLEL = $(patsubst %.o, %.d, $(OFILES_PARALLEL))
DFILES_SERIAL = $(patsubst %.o, %.d, $(OFILES_SERIAL))

all: $(TARGET_PARALLEL) $(TARGET_SERIAL)

clean:
	@echo cleaning up...
	@rm -rf $(ODIR)

test:
	@cat datasets/dataset3.txt | $(TARGET_PARALLEL)

open:
	@vim -p kmeans_serial.cpp kmeans_parallel.cpp Makefile README.txt instructions.txt datasets/dataset1.txt datasets/dataset2.txt

$(ODIR)/kmeans_parallel.o: kmeans_parallel.cpp
	@echo [CXX] $< "-->" $@
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

$(ODIR)/kmeans_serial.o: kmeans_serial.cpp
	@echo [CXX] $< "-->" $@
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET_PARALLEL): $(OFILES_PARALLEL)
	@echo [LD] $^ "-->" $@
	@$(CXX) -o $@ $^ $(LDFLAGS)

$(TARGET_SERIAL): $(OFILES_SERIAL)
	@echo [LD] $^ "-->" $@
	@$(CXX) -o $@ $^ $(LDFLAGS)

# Remember that 'all' and 'clean' aren't real targets
.PHONY: all clean

# Pull in all dependencies
-include $(DFILES_PARALLEL)
-include $(DFILES_SERIAL)

