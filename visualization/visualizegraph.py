import networkx as nx
import matplotlib.pyplot as plt
import csv


g = nx.MultiGraph()

with open("./results/raw/network.csv") as file:
    #header = next(csv.reader(file))
    node1 = 0
    while(True):
        try:
            row = next(csv.reader(file))
        except StopIteration:
            break
        for node2 in row:
            g.add_edge(node1, node2)
        node1 += 1
        
 
nx.draw(g, with_labels = True)
plt.show()
#plt.savefig("filename.png")