import csv
import numpy as np

from scipy.stats import binom

import matplotlib.pyplot as plt

def plotdistribution():
    with open("./data/degreedistribution/power_law_n10000_a25.csv") as file:
        row = next(csv.reader(file))

    row = [int(i) for i in row]

    def powerlawpmf(k, n=10000, alpha=2.5):
        C = sum([ks**(-alpha) for ks in range(1, int(n**0.5)+1)])
        return (k**(-alpha))/C

    data = [0]*(int(10000**0.5)+1)
    for i in row:
        data[i] += 1
        
    data.remove(0)
    c = sum(data)
    data = [i/c for i in data]

    data_clean = [7455,1315,481,233,131,85,65,53,15,30,24,13,8,8,11,13,5,3,0,8,6,3,1,4,2,0,3,1,4,1,3,2,0,2,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]

    c_clean = sum(data_clean)
    data_clean = [i/c_clean for i in data_clean]

    mean = sum([(i+1)*data[i] for i in range(0, len(data))])
    print(mean)
    mean_clean = sum([(i+1)*data_clean[i] for i in range(0, len(data_clean))])
    print(mean_clean)

    truth = [powerlawpmf(k) for k in range(1, int(10000**0.5)+1)]

    diff = [i-j for i, j in zip(data, truth)]
    diff_clean = [i-j for i, j in zip(data_clean, truth)]

    plt.plot(range(1, 101), diff[:100], 'x', fillstyle='none', label='original sequence minus theoretical power-law')
    plt.plot(range(1, 101), diff_clean[:100], 'x', fillstyle='none', label = 'after self/multi edge removal minus theoretical power-law')

    #plt.plot(range(1, 11), data[:10], 'o', fillstyle='none', label = 'sequence distribution')#, linestyle='none')
    #plt.plot(range(1, 11), data_clean[:10], 's', fillstyle='none', label = 'after self/multi edge removal')#, linestyle='none')
    #plt.plot(range(1, 11), truth[:10], 'x', fillstyle='none', label =  'theoretical power-law')#, linestyle='none')
    plt.xlabel("degree k")
    plt.ylabel("difference of probability")
    plt.title("Degree distribution")
    plt.grid()
    plt.legend()
    plt.show()


def plotneighborsdegree():
    #with open("./data/degreedistribution/excessdegree.csv") as file:
    #    row = next(csv.reader(file))

    #row = [0 if i=='0' else float(i) for i in row]

    row = [7.67779,7.55062,7.53789,7.36237,7.25102,7.14672,6.9226,6.93125,6.79561,6.58817,6.35685,6.07566,6.06355,6.07141,5.77936,5.90944,5.3259,5.59207,5.29421,5.09808,4.64802,4.68246,4.47373,4.64285,4.83529,4.894,4.58453,3.52617,4.3375,4.30277,3.94822,3.77164,3.71099,4.07764,4.68238,3.94187,3.44196,4.02632,3.72494,3.72524,3.45788,3.87958,3.82773,3.77409,4.36231,4.00048,3.33481,3.21651,3.83636,3.15356,3.26294,4.43234,3.06653,3.5853,2.94161,3.61646,2.91292,3.68691,2.61461,3.01333,3.60155,3.59892,3.25272,3.27897,2.88615,3.23627,3.38691,2.78204,2.84879,3.06333,3.71643,2.68269,2.82406,2.65904,3.10571,3.06842,3.39574,3.49908,3.00181,3.17344,2.93827,3.31382,3.59639,2.91815,2.45752,3.56131,2.58621,2.93182,3.02622,2.55648,2.83987,2.3587,2.57527,2.64716,2.05614,2.54514,2.06959,2.80612,2.5202]
    #row = [8.50568,7.96256,7.26795,6.97352,7.96604,8.95417,7.57366,4.43966,4.16846,7.53,5.16818,7.08333,3.77692,6.625,9.42222,4.125,1.41176,5.03704,6.04678,4.08333,3.87755,4.95455,4.40217,4.41667,0,9.88462,1.66667,1.96429,0,0,5.96774,0,0,1.52941,3.5619,2.19444,0,3.39474,4.08974,4.475,0,5.42857,0,2.88636,0,0,1.92553,0,0,0,0,0,0,0,0,0,3.68421,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.09211,0,2.07692,0,0,0,0,0,0,0,2.63953,0,0,0,0,2.18681,0,0,0,0,0,0,0,0]
    k2_k = [7.05989]#[7.38681]#;

    plt.plot(range(1, int(10000**0.5)), row, 'o', fillstyle='none', linestyle='none', label='simulation data')
    plt.plot(range(1, int(10000**0.5)), k2_k*len(row), label='<k^2>/<k>')
    plt.legend()
    plt.ylim([1, 100])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('degree k')
    plt.ylabel('knn(k)')
    plt.title("Neighbors' average degree - scale free n=10000, alpha=2.5, runs=100")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    plotdistribution()
    #plotneighborsdegree()