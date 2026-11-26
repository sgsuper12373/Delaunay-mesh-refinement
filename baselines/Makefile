CC ?= gcc
CFLAGS ?= -O2
all: bin/triangle bin/bin2poly
bin/triangle: triangle/triangle.c | bin
	$(CC) $(CFLAGS) -std=gnu89 -w -DNO_TIMER -DLINUX -o $@ $< -lm
bin/bin2poly: bin2poly.c | bin
	$(CC) $(CFLAGS) -o $@ $<
bin:
	mkdir -p bin
test: all
	bin/triangle -pq20 tests/A.poly
clean:
	rm -f bin/triangle bin/bin2poly
