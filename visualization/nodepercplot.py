import csv
import numpy as np

from scipy import optimize
from scipy.stats import binom

import matplotlib.pyplot as plt

import yaml


def binomialanalyticalsolution(phy, n, p, upper_limit=50):
    #print(binom.pmf(1, n, p))
    def fun(u): #15.3 15.4 15.5
        return u-1+phy-(phy/(n*p))*sum([(k+1)*binom.pmf(k+1, n, p)*u**k for k in range(0, upper_limit)])

    sol = optimize.root(fun, 0.5)

    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    return phy*(1-sum([binom.pmf(k, n, p)*sol**k for k in range(0, upper_limit)])) #15.2

def main():
    with open("./experiments/config.yaml") as file:
        config_params = yaml.load(file, Loader=yaml.FullLoader)
        #print(config_params)

    for exp_n, exp_params in config_params['giant_component'].items():
        with open("./results/raw/node_perc_giant_cluster_exp_{}.csv".format(exp_n)) as file:
            #header = next(csv.reader(file))
            row = next(csv.reader(file))

        row = [float(i)/int(exp_params['network_size']) for i in row] #convert to float and normalize
        if exp_params['network_type'] == 'u':
            legend = "n={}, runs={}; degree dist.: U({}, {})".format(exp_params['network_size'],
                                                                     exp_params['runs'],
                                                                     exp_params['param1'],
                                                                     exp_params['param2'])
        elif exp_params['network_type'] == 'b':
            legend = "n={}, runs={}; degree dist.: Bin({}, {})".format(exp_params['network_size'],
                                                                       exp_params['runs'],
                                                                       exp_params['param1'],
                                                                       exp_params['param2'])
        elif exp_params['network_type'] == 'p':
            legend = "n={}, runs={}; degree dist.: p({})".format(exp_params['network_size'],
                                                                 exp_params['runs'],
                                                                 exp_params['param1'])


        x = np.arange(0, 1, 1/len(row))
        plt.plot(x, row, label=legend)

        truth = [binomialanalyticalsolution(i, 1000, 0.003) for i in x]
        plt.plot(x, truth, label="Analytical solution")

        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.title("Average size of the largest cluster as a function of φ")
        plt.legend(loc='upper left')
        plt.xlabel("Occupation probability φ")
        plt.ylabel("Size of giant cluster S(φ)")
        plt.grid(True)
        #plt.show()
        plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
        plt.close()


if __name__ == '__main__':
    main()