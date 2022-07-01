import csv
import numpy as np

from scipy import optimize
from scipy.stats import binom

import matplotlib.pyplot as plt

import yaml

def targetedbinomialanalyticalsolution(k0, n, p):
    #sum_phy = 0
    #k = 0
    #while(sum_phy <= phy):
    #    sum_phy += binom.pmf(k, n, p)
    #    k += 1
    #k0 = k-1

    def f0(z):
        return sum([binom.pmf(k, n, p)*z**k for k in range(0, k0)])

    def f1(z):
        return 1/(n*p) * sum([k*binom.pmf(k, n, p) * z**(k-1) for k in range(1, k0)])

    def fun(u):
        return u - 1 + f1(1) -f1(u)

    sol = optimize.root(fun, 0.5)
    #print(k0, sol.x)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    return f0(1)-f0(sol)

def binomialanalyticalsolution(phy, n, p, upper_limit=50):
    #print(binom.pmf(1, n, p))
    def fun(u): #15.3 15.4 15.5
        return u-1+phy-(phy/(n*p))*sum([(k+1)*binom.pmf(k+1, n, p)*(u**k) for k in range(0, upper_limit)])

    sol = optimize.root(fun, 0.5)
    #print(phy, sol.success)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    return phy*(1-sum([binom.pmf(k, n, p)*(sol**k) for k in range(0, upper_limit)])) #15.2

def bondbinomialanalyticalsolution(phy, n, p, upper_limit=50):
    #print(binom.pmf(1, n, p))
    def fun(u): #15.3 15.4 15.5
        return u-1+phy-(phy/(n*p))*sum([(k+1)*binom.pmf(k+1, n, p)*(u**k) for k in range(0, upper_limit)])

    sol = optimize.root(fun, 0.5)
    #print(phy, sol.success)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    return (1-sum([binom.pmf(k, n, p)*(sol**k) for k in range(0, upper_limit)]))

def powerlawanalyticalsolution(phy, alpha, n, min_cutoff=2):
    #print(phy)

    C = sum([ks**(-alpha) for ks in range(min_cutoff, int(n**0.5)+1)])

    def powerlawpmf(k):
        return (k**(-alpha))/C

    mean = sum([k*powerlawpmf(k) for k in range(min_cutoff, int(n**0.5)+1)])
    #print(n, mean)
    
    def g1(u):
        return sum([(k+1)*powerlawpmf(k+1)*(u**k) for k in range(min_cutoff-1, int(n**0.5))])/mean

    def fun(u): #15.3 15.4 15.5
        return 1-phy+(phy)*g1(u) - u

    sol = optimize.root(fun, 0)

    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    def g0(u):
        return sum([powerlawpmf(k)*(u**k) for k in range(min_cutoff, int(n**0.5)+1)])
    return phy*(1-g0(sol)) #15.2

def analyticalSolutionsComparison():
    x = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    exp = [21.79/100, 420.93/1000, 5011.12/10000, 52596.1/100000]
    exp_bfs = [21.79/100, 420.93/1000, 5011.12/10000, 52596.1/100000]
    real = [powerlawanalyticalsolution(1, 2.5, i) for i in x[:4]]
    real_sqrt = [powerlawanalyticalsolution(1, 2.5, i**0.5) for i in x]
    plt.plot(x[:len(exp)], exp, marker='o', fillstyle='none', label="simulation result (Newman)")
    plt.plot(x[:len(exp_bfs)], exp_bfs, marker='o', fillstyle='none', label="simulation result (BFS)")
    plt.plot(x, real_sqrt, marker='o', fillstyle='none', label="analytical solution cutoff=sqrt(n)")
    plt.plot(x[:4], real, marker='o', fillstyle='none', label="analytical solution cutoff=n")

    plt.xscale('log')
    plt.legend()
    plt.xlabel("number of nodes")
    plt.ylabel("Size of giant cluster S(φ=1)")
    plt.grid()
    plt.title("analytical vs simulation: scale free alpha=2.5, φ=1, runs=100")
    plt.show()

def targetedattackplot(exp_n, exp_params, row):
    while(row[-1] == 0.0):
        row.pop()
    x = range(0, len(row))

    legend = "n={}, runs={};\ndegree dist.: Bin({}, {})".format(exp_params['network_size'],
                                                            exp_params['runs'],
                                                            exp_params['param1'],
                                                            exp_params['param2'])
    truth = [targetedbinomialanalyticalsolution(k0, exp_params['param1'], exp_params['param2']) for k0 in x]
    plt.plot(x, truth, color='#ff7f0e', marker='o', fillstyle='none', label="Analytical solution")
    plt.plot(x, row, color='#1f77b4', marker='x', fillstyle='none', linestyle='none', label=legend)
    #plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.title("Average size of the largest cluster\nas a function of the maximum degree k0 (targeted attack)")
    plt.legend(loc='lower right')
    plt.xlabel("Maximum degree k0")
    plt.ylabel("Size of giant cluster S")
    plt.grid(True)
    #plt.show()
    plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
    plt.close()

def linkpercolation(exp_n, exp_params, row):
    x = np.arange(0, 1+1/len(row), 1/(len(row)-1))
    legend = "n={}, runs={};\ndegree dist.: Bin({}, {})".format(exp_params['network_size'],
                                                                       exp_params['runs'],
                                                                       exp_params['param1'],
                                                                       exp_params['param2'])
    truth = [bondbinomialanalyticalsolution(i, exp_params['param1'], exp_params['param2']) for i in x]
    truth2 = [binomialanalyticalsolution(i, exp_params['param1'], exp_params['param2']) for i in x]

    plt.plot(x, row, marker='o', fillstyle='none', linestyle='none', label=legend)
    plt.plot(x, truth, label="Analytical sol. - bond perc.")
    plt.plot(x, truth2, linestyle='dashed', label="Analytical sol. - site perc.")

    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.title("Average size of the largest cluster as a function of φ")
    plt.legend(loc='lower right')
    plt.xlabel("Occupation probability φ")
    plt.ylabel("Size of giant cluster S(φ)")
    plt.grid(True)
    #plt.show()
    plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
    plt.close()

def main():
    with open("./experiments/test.yaml") as file:
        config_params = yaml.load(file, Loader=yaml.FullLoader)
        #print(config_params)

    for exp_n, exp_params in config_params['giant_component'].items():
        with open("./results/raw/node_perc_giant_cluster_exp_{}.csv".format(exp_n)) as file:
            #header = next(csv.reader(file))
            row = next(csv.reader(file))

        x = np.arange(0, 1+1/len(row), 1/(len(row)-1))
        row = [float(i)/int(exp_params['network_size']) for i in row] #convert to float and normalize
        if exp_params['percolation_type'] == 't':
            targetedattackplot(exp_n, exp_params, row)
            return 

        if exp_params['percolation_type'] == 'l':
            linkpercolation(exp_n, exp_params, row)
            return

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
            truth = [binomialanalyticalsolution(i, exp_params['param1'], exp_params['param2']) for i in x]
        elif exp_params['network_type'] == 'p':
            legend = "n={}, runs={}; degree dist.: p({})".format(exp_params['network_size'],
                                                                 exp_params['runs'],
                                                                 exp_params['param1'])
            truth = [powerlawanalyticalsolution(i, exp_params['param1'], exp_params['network_size']) for i in x]

        plt.plot(x, row, marker='o', fillstyle='none', linestyle='none', label=legend)
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