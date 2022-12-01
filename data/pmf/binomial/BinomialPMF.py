from scipy.stats import binom
import numpy as np
import csv

nodes = 10000
bins = 21
# open the file in the write mode
with open('./data/pmf/binomial/binomialpmf_n{}_b{}.csv'.format(nodes, bins), 'w', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)

    for phy in np.arange(0, 1+1/bins, 1/(bins-1)):
        print(phy)
        row = []
        for k in range(0, nodes+1):
            prob = binom.pmf(k, nodes, phy)
            if prob < 1e-300:
                prob = 0
            row.append(prob)

        # write a row to the csv file
        writer.writerow(row)