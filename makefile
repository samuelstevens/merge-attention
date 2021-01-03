CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-overflow -fpic

attention/attentionmodule.so: build/attentionmodule.o
	$(CC) -shared $? -o $@ 

build/attentionmodule.o: src/attentionmodule.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $? -o $@

clean:
	rm -f *.so *.o
	rm -rf build/
	rm -rf dist/
	rm -rf attention.egg-info
	rm attention/attentionmodule.so
