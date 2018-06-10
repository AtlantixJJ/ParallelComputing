import os

Ns = [4000, 8000, 12000, 16000, 20000, 32000, 44000]

os.system("rm log.txt")

for proc_num in [4, 10, 20, 40]:
    for N in Ns:
        for i in range(5):
            print("mpirun -np %d build/prob1 %d 1" % (proc_num, N))
            os.system("mpirun -np %d build/prob1 %d 1" % (proc_num, N))