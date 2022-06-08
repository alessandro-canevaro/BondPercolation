import csv
import numpy as np
import matplotlib.pyplot as plt

with open("./results/node_perc_giant_cluster.csv") as file:
    row = next(csv.reader(file))

row = [float(i) for i in row]
plt.plot(np.arange(0, 1, 1/len(row)), row)
plt.xlim((-0.1, 1.1))
plt.show()