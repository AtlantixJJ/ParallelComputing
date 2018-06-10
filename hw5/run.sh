rm parallel_y serial_y
mpirun -np 4 build/new 8
diff parallel_y serial_y