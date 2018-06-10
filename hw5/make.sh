mkdir build

#mpicc src/test.c -o build/test

#mpicc -O3 -g -Wall -o build/prob1 src/prob1.c

gcc -g -Wall src/prob2.c -o build/prob2 -lpthread

gcc -g -Wall -std=c++11 -fopenmp src/prob3.c -o build/prob3

gcc -g -Wall -std=c++11 -fopenmp src/prob3_a.c -o build/prob3_a

gcc -g -Wall -std=c++11 -fopenmp src/prob3_b.c -o build/prob3_b

gcc -g -Wall -std=c++11 -fopenmp src/prob4.c -o build/prob4 -lpthread

#mpicc -g -Wall -o build/prog3.5_mpi_mat_vect_col src/prog3.5_mpi_mat_vect_col.c