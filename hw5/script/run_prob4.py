import os
os.system("rm log_prob3.txt")

Ns = [4000, 8000, 12000, 16000, 20000, 32000, 44000]

for proc_num in [4, 10, 20, 40]:
    for N in Ns:
        for i in range(5):
            print("build/prob4 %d %d" % (N, proc_num))
            os.system("build/prob4 %d %d" % (N, proc_num))