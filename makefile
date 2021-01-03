CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-overflow -fpic

attentionmodule.so: attentionmodule.o
	$(CC) -shared $? -o $@ 

attentionmodule.o: attentionmodule.c
	$(CC) $(CFLAGS) -c $? -o $@

clean:
	rm -f *.so *.o
	rm -rf build/
