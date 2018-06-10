import matplotlib.pyplot as plt
import numpy as np

filelines = open("log_prob4.txt").readlines()

Ns = [4000, 8000, 16000, 32000, 44000]
NPs = [4, 10, 20, 40]

t1 = []
t2 = []
t3 = []
t4 = []

for l in filelines:
    x = l.strip().split(",")
    if len(x) != 4:
        continue

    t1.append(float(x[0]))
    t2.append(float(x[1]))
    t3.append(float(x[2]))
    t4.append(float(x[3]))

t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)
t4 = np.array(t4)
print(t1.shape)

stime = []
for i in range(5):
    stime.append(t1[i::5].mean())
stime = np.array(stime)
np.save("baseline1", stime)
#stime = np.load("baseline1.npy")
#print(stime)

# execution time
proc_exec_time1 = []
proc_exec_time2 = []
proc_exec_time3 = []
for i in range(len(NPs)):
    local_array = [t2[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)]
    proc_exec_time1.append(np.array(local_array))

    local_array = [t3[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)]
    proc_exec_time2.append(np.array(local_array))

    local_array = [t4[i*35+j*5:i*35+(j+1)*5].mean() for j in range(5)]
    proc_exec_time3.append(np.array(local_array))

for i in range(len(NPs)):
    plt.plot(Ns, stime)
    plt.plot(Ns, proc_exec_time1[i])
    plt.plot(Ns, proc_exec_time2[i])
    plt.plot(Ns, proc_exec_time3[i])
    plt.legend(["serial", "static p=%d" % NPs[i], "dynamic p=%d" % NPs[i], "pthread p=%d" % NPs[i]])
    ax = plt.gca()
    ax.set_ylabel("miliseconds")
    ax.set_xlabel("N")
    plt.savefig("fig/exectime_2_%d.png" % i)
    plt.close()

for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time1[i])
    plt.plot(Ns, proc_exec_time2[i])
    plt.plot(Ns, proc_exec_time3[i])
    plt.legend(["static p=%d" % NPs[i], "dynamic p=%d" % NPs[i], "pthread p=%d" % NPs[i]])
    ax = plt.gca()
    ax.set_ylabel("miliseconds")
    ax.set_xlabel("N")
    plt.savefig("fig/c_exectime_2_%d.png" % i)
    plt.close()

legends = []
for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time1[i], 'r')
    legends.append("static p=%d" % NPs[i])
for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time2[i], 'c')
    legends.append("dynamic p=%d" % NPs[i])
for i in range(len(NPs)):
    plt.plot(Ns, proc_exec_time3[i], 'b')
    legends.append("pthread p=%d" % NPs[i])

plt.legend(legends)
ax = plt.gca()
ax.set_ylabel("miliseconds")
ax.set_xlabel("N")
plt.savefig("fig/total_2.png")
plt.close()