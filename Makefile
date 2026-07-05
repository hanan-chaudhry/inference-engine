CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fPIC -fvisibility=default
LDFLAGS = -shared
INCLUDES = -Iinc -I/usr/local/include
LIBS = -L/usr/local/lib -lblis -lm

SRC_DIR = src/kernals
BUILD_DIR = build
TEST_DIR = tests/unit

SOURCES = $(wildcard $(SRC_DIR)/*.c)
TARGET = $(BUILD_DIR)/libkernels.so
TEST_RUNNER = tests/run_tests.py

.PHONY: all build test clean

all: build

build: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

test: build
	@python $(TEST_RUNNER)

clean:
	rm -rf $(BUILD_DIR)