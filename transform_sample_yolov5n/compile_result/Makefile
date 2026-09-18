
MAKEFLAGS += -s

CROSS_COMPILE:=/home/alex/Downloads/magik-toolkit-v3.0/TransformKit/mips-gcc540-glibc222-r3.3.7.mxu2.cve/bin/mips-linux-gnu-
CC = $(CROSS_COMPILE)gcc
CPP = $(CROSS_COMPILE)g++
AR = $(CROSS_COMPILE)gcc-ar
STRIP = $(CROSS_COMPILE)strip
LIBTYPE ?= muclibc

HOME_LIB_PATH = ./tmp_lib


DEF_OPTTIONS= -std=c++11 -mfp32 -Wall -EL -O3 -Os -march=mips32r2 -flax-vector-conversions -lpthread -lrt -ldl -lm -fPIC -flto -Wl,-Map=object.map,--cref,--gc-section

CXXFLAGS := $(DEFEINE_OPT) -ffunction-sections -fdata-sections -fno-exceptions

ifeq ($(LIBTYPE), muclibc)
    CXXFLAGS += -muclibc
	LIB_PATH = $(HOME_LIB_PATH)/540/lib/uclibc
else
	LIB_PATH = $(HOME_LIB_PATH)/540/lib/glibc
endif

# THIRD_PATH=-L$(LIB_PATH) -laip -L$(LIB_PATH) -ldrivers
INCLUDES := -I$(HOME_LIB_PATH) -I$(HOME_LIB_PATH)/include

TARGET = load_mgk_public

all : $(TARGET)

$(TARGET):main.o
	$(CPP) $(CXXFLAGS) $^ -o $@ $(INCLUDES) $(LIB_PATH)/liburanus_tensor.so $(LIB_PATH)/liburanus_core.so $(LIB_PATH)/libaie.so -Wl,-no-whole-archive -Wl,-gc-sections -fno-exceptions -Wl,-export-dynamic

main.o:main.cpp
	$(CPP) $(DEF_OPTTIONS) $(INCLUDES) $(CXXFLAGS) -o $@ $^ -c -ffunction-sections -fdata-sections -fno-exceptions

clean:
	rm -rf main.o $(MODEL_O) $(TARGET) $(MODEL_LIB)

.PHONY: all clean

