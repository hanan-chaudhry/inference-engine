CC = gcc

CFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lblis -lm

TARGET = kernels.so
SRC = attention.o softmax.o

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -shared -o $(TARGET) $(SRC) $(LDFLAGS)

softmax.o: softmax.c
	$(CC) -c softmax.c $(CFLAGS) -o softmax.o

attention.o: attention.c
	$(CC) -c attention.c $(CFLAGS) -o attention.o

clean:
	rm -f $(SRC) $(TARGET)