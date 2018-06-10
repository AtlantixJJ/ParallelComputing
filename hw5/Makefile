target : main.o
	gcc -o target main.o

main.o : main.c gen.h
	gcc -c main.c

clean :
	rm target main.o