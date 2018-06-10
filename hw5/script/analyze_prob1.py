import matplotlib.pyplot as plt
import numpy as np

filelines = open("fig/log_prob1.txt").readlines()

Ns = [4000, 8000, 16000, 32000, 44000]
NPs = [4, 10, 20, 40]

t1 = []
t2 = []
t3 = []

for l in filelines:
    x = l.strip().split(",")
    if len(x) != 5:
        continue

    t1.append(float(x[0]))
    t2.append(float(x[1]))
    t3.append(float(x[2]))

t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)

print(t1.shape)

stime = 0
for i in range(len(NPs)):
    tmp = np.array([t2[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)])
    print(tmp)
    stime += tmp
stime = stime / 5
np.save("baseline1", stime)
#stime = np.load("baseline1.npy")
#print(stime)

# execution time
proc_exec_time = []
proc_exec_time_l2 = []
plt.plot(Ns, stime)
for i in range(len(NPs)):
    local_array = [t1[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)]
    local_array = [t3[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)]
    proc_exec_time_l2.append(np.array(local_array))
    proc_exec_time.append(np.array(local_array))
    plt.plot(Ns, proc_exec_time[-1])
proc_exec_time = np.array(proc_exec_time)
plt.legend(["p=1", "p=4", "p=10", "p=20", "p=40"])
ax = plt.gca()
ax.set_ylabel("miliseconds")
ax.set_xlabel("N")
plt.savefig("fig/exectime_1_large.png")
plt.close()

for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time_l2[i])
plt.legend(["p=4", "p=10", "p=20", "p=40"])
ax = plt.gca()
ax.set_ylabel("miliseconds")
ax.set_xlabel("N")
plt.savefig("fig/exectime_1_l2.png")
plt.close()

for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time[i])
plt.legend(["p=4", "p=10", "p=20", "p=40"])
ax = plt.gca()
ax.set_ylabel("miliseconds")
ax.set_xlabel("N")
plt.savefig("fig/exectime_1_small.png")
plt.close()

# speedup
spd_up = []
for i in range(len(NPs)):
    spd_up.append(stime / proc_exec_time[i])
    plt.plot(Ns, spd_up[-1])
spd_up = np.array(spd_up)
plt.legend(["p=4", "p=10", "p=20", "p=40"])
ax = plt.gca()
ax.set_ylabel("ratio")
ax.set_xlabel("N")
plt.savefig("fig/speedup_1.png")
plt.close()

# efficiency
for i in range(len(NPs)):
    plt.plot(Ns, spd_up[i] / NPs[i] * 100)
plt.legend(["p=4", "p=10", "p=20", "p=40"])
ax = plt.gca()
ax.set_ylabel("percent")
ax.set_xlabel("N")
plt.savefig("fig/efficiency_1.png")
plt.close()
