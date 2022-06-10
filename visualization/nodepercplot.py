import csv
import numpy as np
import matplotlib.pyplot as plt

with open("./results/node_perc_giant_cluster.csv") as file:
    header = next(csv.reader(file))
    row = next(csv.reader(file))

row = [float(i)/int(header[1]) for i in row]

plt.plot(np.arange(0, 1, 1/len(row)), row, label=str(header))
plt.xlim((-0.1, 1.1))
plt.ylim((-0.1, 1.1))
plt.title("Average size of the largest cluster as a function of φ")
plt.legend(loc='upper left')
plt.xlabel("Occupation probability φ")
plt.ylabel("Size of giant cluster S(φ)")
plt.grid(True)
plt.show()