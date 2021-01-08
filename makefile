CC = gcc
CFLAGS = -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-overflow -fpic

attention/attentionmodule.so: build/attentionmodule.o
	$(CC) -shared $? -o $@ 

build/attentionmodule.o: src/attentionmodule.c
	mkdir -p build
	$(CC) $(CFLAGS) -c $? -o $@

clean: FORCE
	rm -f *.so *.o
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -f .coverage
	cargo clean

test: FORCE
	python -m pytest --cov=attention tests/

FORCE:
