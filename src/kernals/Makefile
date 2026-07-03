CC = gcc

CFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lblis -lm

TARGET = main
SRC = main.c softmax.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) $(CFLAGS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)