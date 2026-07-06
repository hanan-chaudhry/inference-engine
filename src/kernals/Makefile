CC = gcc

CFLAGS = -I/usr/local/include
LDFLAGS = -L/usr/local/lib -lblis -lm

TARGET = kernals.so
SRC = attention.c softmax.c multiAttention.c relu.c sigmoid.c silu.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) -shared -o $(TARGET) $(SRC) $(LDFLAGS)

softmax.so: softmax.c
	$(CC) -shared -fPIC softmax.c $(CFLAGS) $(LDFLAGS) -o softmax.so

attention.so: attention.c softmax.c
	$(CC) -shared -fPIC attention.c softmax.c $(CFLAGS) $(LDFLAGS) -o attention.so

layernorm.so: layernorm.c
	$(CC) -shared -fPIC layernorm.c $(CFLAGS) $(LDFLAGS) -o layernorm.so
	
clean:
	rm -f $(TARGET)