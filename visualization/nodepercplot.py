import csv
from importlib.machinery import PathFinder
from tkinter import W
from unittest import result
import numpy as np

from scipy import optimize
from scipy.stats import binom, poisson
from scipy.misc import derivative
from scipy.special import factorial
from scipy import linalg
from scipy.sparse import linalg
from sympy import Matrix

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
        #return u-1+phy-phy*u**2
        return u-1+phy-(phy/(n*p))*sum([(k+1)*binom.pmf(k+1, n, p)*(u**k) for k in range(0, upper_limit)])

    sol = optimize.root(fun, 0.5)
    #print(phy, sol.success)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError

    #return 1 - sol**3
    return (1-sum([binom.pmf(k, n, p)*(sol**k) for k in range(0, upper_limit)]))

def test(F0, n, p, upper_limit=50, mu=8):
    phi = sum([binom.pmf(f, 20, 0.5) for f in range(0, F0)])
    m = sum([binom.pmf(f, 20, 0.5)*f for f in range(0, upper_limit)])

    def g0(z):
        return sum([z**(k) * binom.pmf(k, n, p) for k in range(0, upper_limit)])

    def g1(z):
        return sum([z**(k-1) *k* binom.pmf(k, n, p) for k in range(1, upper_limit)]) / (n*p)

    def fun(u):
        return u - 1 + phi*(g1(1) - g1(u))
        #return 1 - g1(1-u) - u

    sol = optimize.root(fun, 0.5)
    #print(F0, sol.x)
    if sol.success:
        sol = sol.x[0]
    else:
        sol= 0
        print("error")
        #raise AssertionError

    
    print(F0, phi, sol, g0(1), g0(sol))
    if phi == 0:
        return 0
    return (g0(1) - g0(sol))

def featurebondanalyticalsolution(F0, n, p, upper_limit=50, mu=8):
    phi = sum([poisson.pmf(f, 8) for f in range(0, F0)])
    #print(binom.pmf(1, n, p))
    def g0(z):
        return sum([binom.pmf(k, n, p)*(z**k) for k in range(0, upper_limit)]) 

    mean = sum([poisson.pmf(f, 8) * sum([binom.pmf(k, n, p)*k for k in range(0, upper_limit)]) for f in range(0, 50)])

    def g1(z):
        return sum([binom.pmf(k, n, p)*k*z**(k-1) for k in range(1, upper_limit)])  / (n*p)

    def fun(u):
        return u - 1 + g1(1-phi*u)
        return u - 1 + phi - phi*g1(u)
        #return 1 - g1(1-u) - u

    sol = optimize.root(fun, 0.5)
    #print(F0, sol.x)
    if sol.success:
        sol = sol.x[0]
    else:
        raise AssertionError

    #print(g0(1-sol))
    #return g0(1) - g0(sol)
    #print(f0(1), f0(sol))

    if phi == 0:
        return 0
    return 1 - g0(1-phi*sol)
    return (1 - g0(sol)) 

def powerfeaturebondanalyticalsolution(F0, n, alpha, upper_limit=50, mu=8, min_cutoff=2):
    phi = sum([poisson.pmf(f, 8) for f in range(0, F0)])

    C = sum([ks**(-alpha) for ks in range(min_cutoff, int(n**0.5)+1)])

    def powerlawpmf(k):
        return (k**(-alpha))/C

    #print(binom.pmf(1, n, p))
    def g0(z):
        return sum([powerlawpmf(k)*(z**k) for k in range(min_cutoff, int(n**0.5)+1)]) 

    mean = sum([k*powerlawpmf(k) for k in range(min_cutoff, int(n**0.5)+1)])

    def g1(z):
        return sum([(k+1)*powerlawpmf(k+1)*(z**k) for k in range(min_cutoff-1, int(n**0.5))])/mean

    def fun(u):
        #return u - 1 + g1(1-phi*u)
        return u - 1 + phi - phi*g1(u)
        #return 1 - g1(1-u) - u

    sol = optimize.root(fun, 0.5)
    print(F0, sol.x)
    if sol.success:
        sol = sol.x[0]
    else:
        raise AssertionError

    #print(g0(1-sol))
    #return g0(1) - g0(sol)
    #print(f0(1), f0(sol))

    if phi == 0:
        return 0
    #return 1 - g0(1-phi*sol)
    return (1 - g0(sol)) 

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

def featurebondperc(exp_n, exp_params, row):
    x = np.arange(0, 20)

    #truth = [featurebondanalyticalsolution(i, exp_params['param1'],  exp_params['param2'], 50, 8) for i in x] #ER
    truth = [powerfeaturebondanalyticalsolution(i, exp_params['network_size'], exp_params['param1']) for i in x] #SF

    plt.plot(x, row, marker='x', fillstyle='none', linestyle='dashed', label="Simulation (runs={})".format(exp_params['runs']))
    plt.plot(x, truth, marker='o', fillstyle='none', label="Analytical sol. - bond perc.")
    #plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    #plt.title("Average size of the largest cluster \n n={}, degree dist.: Bin(n={}, p={})".format(exp_params['network_size'], #ER
    #                                                                                            exp_params['param1'],
    #                                                                                            exp_params['param2']))
    plt.title("Average size of the largest cluster \n n={}, degree dist.: Power-law, alpha={}".format(exp_params['network_size'],
                                                                                                    exp_params['param1'])) #SF                                                                                        
                                                                                        
    plt.legend(loc='upper left')
    plt.xlabel("Feature F: Poisson(mu=8)")
    plt.ylabel("Size of giant cluster S(φ)")
    plt.grid(True)
    #plt.show()
    plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
    plt.close()

def cavitysolution(F0, n, p, upper_limit=10):
    
    def pk(k):
        return binom.pmf(k, n, p)

    def pfkk(f, m):
        return poisson.pmf(f, m)

    mean = n*p

    def phi(f):
        if f >= F0:
            return 0
        return 1

    def vecfun(u):
        result = np.zeros_like(u)
        for k in range(1, upper_limit):
            tmp = sum([sum([m*pk(m)/mean * pfkk(f, 50//(m+k)) * phi(f) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, 20)])
            result[k-1] = (1-tmp)**(k-1)
        return u-result

    sol = optimize.root(vecfun, np.zeros((upper_limit-1, 1))+0.0, method='lm')
    if not sol.success:
        print("ERROR")

    W = np.zeros((upper_limit, 1))
    W[0] = 1
    for k in range(1, upper_limit):
        tmp = sum([sum([m*pk(m)/mean * pfkk(f, 50//(m+k)) * phi(f) * (1-sol.x[m-1]) for m in range(1, upper_limit)]) for f in range(0, 20)])
        W[k] = (1-tmp)**k

    S = sum([pk(k)*(1-W[k]) for k in range(0, upper_limit)])
    print(F0, S)
    return S

def corrfeaturebondperc(exp_n, exp_params, row):
    x = np.arange(0, 20)

    truth = [cavitysolution(i, exp_params['param1'],  exp_params['param2']) for i in x] #ER
    
    plt.plot(x, row[0:x[-1]+1], marker='x', fillstyle='none', linestyle='dashed', label="Simulation (runs={})".format(exp_params['runs']))
    plt.plot(x, truth, marker='o', fillstyle='none', label="Analytical sol. - bond perc.")

    plt.ylim((-0.1, 1.1))
    plt.title("Average size of the largest cluster \n n={}, degree dist.: Bin(n={}, p={})".format(exp_params['network_size'], #ER
                                                                                                exp_params['param1'],
                                                                                                exp_params['param2']))
    #plt.title("Average size of the largest cluster \n n={}, degree dist.: Power-law, alpha={}".format(exp_params['network_size'],
    #                                                                                                exp_params['param1'])) #SF                                                                                        
                                                                                        
    plt.legend(loc='lower right')
    plt.xlabel("Feature F: P(F | k, k') = Poisson(50 // (k+k'))")
    plt.ylabel("Size of giant cluster S")
    plt.grid(True)
    #plt.show()
    plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
    plt.close()

def smallsiteanalyticalsolution(phi, n, p, upper_limit=50):
    def g0(z):
        return sum([binom.pmf(k, n, p)*z**k for k in range(0, upper_limit)])

    def g0p(z):
        return sum([binom.pmf(k, n, p)*k*z**(k-1) for k in range(1, upper_limit)])

    def g0s(z):
        return sum([binom.pmf(k, n, p)*k*(k-1)*z**(k-2) for k in range(2, upper_limit)])

    def g1(z):
        return sum([binom.pmf(k, n, p)*k*z**(k-1) for k in range(1, upper_limit)]) / (n*p)

    def g1p(z):
        return sum([binom.pmf(k, n, p)*k*(k-1)*z**(k-2) for k in range(2, upper_limit)]) / (n*p)

    def fun(u):
        return u - 1 + phi - phi*g1(u)

    sol = optimize.root(fun, 0.5)
    #print(phy, sol.success)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError   

    #print(phi, sol, g1p(sol)) 
    #return 1+3*phi

    k_small = (sol*g0p(sol)) / (1-phi*(1-g0(sol)))
    k_n = (sol*g1p(sol)) / g0p(sol)#sol*g0s(sol)/g0p(sol)
    t = 1 / (1+phi*(1-k_n))

    #print(phi, k_small, k_n, t)
    #return (1 + phi*k_small * t)

    def qk(k):
        return (k+1)*binom.pmf(k+1, n, p)/(n*p)
    
    def pk(k):
        return binom.pmf(k, n, p)

    num_p = (1-phi)*sum([k*pk(k) for k in range(0, upper_limit)]) + phi*sum(k*pk(k)*sol**k for k in range(0, upper_limit))

    num_q = (1-phi)*sum([k*qk(k) for k in range(0, upper_limit)]) + phi*sum(k*qk(k)*sol**k for k in range(0, upper_limit))
    print(phi, sol*g0p(sol)/g0(sol), sol*g0s(sol)/g0p(sol))

    S = phi*(1-g0(sol))
    k_small = (3-phi*3+phi*sol*g0p(sol)) / (1-S)#(1-phi*(1-g0(sol)))
    k_big = phi*(3-phi*g0(sol))/S
    #print(phi, k_small, k_big)
    k_n = num_q/(1-S)#sol*g0s(sol)/g0p(sol)#(sol*g1p(sol)) / (1-phi*(1-g1(sol)))
    t = (1) / ((1-k_n))

    #print(phi, k_small, k_n, t)

    return 1/(1-S)*(1 + phi*g0p(1)/(1-phi*g1p(1)))

    return (1 + k_small*(1-S)*t)

def smallcomponent(exp_n, exp_params, row):
    x = np.arange(0, 1+1/len(row), 1/(len(row)-1))
    row = [i*int(exp_params['network_size']) for i in row] 
    truth = [smallsiteanalyticalsolution(phi, exp_params['param1'], exp_params['param2']) for phi in x]

    plt.plot(x, row, marker='o', fillstyle='none', linestyle='none', label="small comp.")
    plt.plot(x, truth, label="Analytical solution")
    #plt.ylim([-1, 10])
    plt.title("Average size of the component to which a small-component\nnode belongs as a function of φ")
    plt.legend(loc='upper right')
    plt.xlabel("Occupation probability φ")
    plt.ylabel("Size of small cluster <s>")
    plt.grid(True)
    #plt.show()
    plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(exp_n))
    plt.close()

def test(phi, s, n, p, upper_limit=50, min_cutoff=0, alpha=3):
    #upper_limit = int(n**0.5)+1
    #C = sum([ks**(-alpha) for ks in range(min_cutoff, upper_limit)])

    def pk(k):
        return (1-0.3)*0.3**(k)#(k**(-alpha))/C#binom.pmf(k, n, p)#(1-0.3)*0.3**k#

    mean = sum([pk(k)*k for k in range(min_cutoff, upper_limit)])

    def G0(z):
        return sum([pk(k)*z**k for k in range(min_cutoff, upper_limit)])
    
    def g0(z):
        return G0(1-phi+phi*z)

    def G1(z):
        return sum([pk(k)*k*z**(k-1) for k in range(min_cutoff+1, upper_limit)]) / mean
    
    def g1(z):
        return G1(1-phi+phi*z)

    def fun2(u):
        return u - 1 + phi - phi*G1(u)

    sol = optimize.root(fun2, 0.5)
    #print(phy, sol.success)
    if sol.success:
        sol = sol.x
    else:
        raise AssertionError 

    S = phi*(1-G0(sol))

    def fun(z):
        return G1(z)**s 


    if(s==1):
        result = phi*sum([pk(k)*(1-phi)**k for k in range(min_cutoff, upper_limit)])
        #result = (1-phi+phi*pk(0))/(1-S)#derivative(G0, 0, n=1)
        print(1, result)
        return result

    #temp = factorial(2*s-1+s-2)/factorial(2*s-1) * g1(0)**s / (0.3**-1)**(s-2)

    result = phi*derivative(fun, 1-phi, n=s-2, dx=1e-2, order=21)*mean*phi**(s-1)/factorial(s-1)
    print(s, result)

    return result

def small_comp_dist(n= 1000000, p = 0.0003):
    
    cut_off = 100
    ax = [1]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1]
    #"""
    #row = [0.69957,0.10484,0.04932,0.02892,0.0181771,0.014087,0.0139103,0.0135294,0.0132453,0.0121053,0.0127742,0.0132857,0.0135417,0.0154,0.0173684,0.0171429,0.0200909,0.02025,0.019,0.0257143,0.021,0.022,0.023,0.024,0.025,0.026,0.027,0.028,0.029,0.03,0.031,0.032,0.033,0.034,0.035,0.036,0.037,0,0.039,0]
    #plt.plot(range(1, cut_off), row, marker='x', fillstyle='none', linestyle='none', label="sim, n=1k, 100 runs")
    #row = [0.700945,0.10199,0.048165,0.027308,0.018145,0.012618,0.009408,0.007376,0.005958,0.00488776,0.00447857,0.00363789,0.00325,0.00308,0.00286875,0.0028274,0.00288136,0.00303934,0.00265208,0.00293878,0.00276818,0.0029,0.00318864,0.00286667,0.00288462,0.00303333,0.00308571,0.00315,0.00322222,0.00375,0.0031,0.00357647,0.00418,0.0034,0.00373333,0.0036,0.0037,0.00456,0.0039,0.004]
    #plt.plot(range(1, cut_off), row, marker='x', fillstyle='none', linestyle='none', label="sim, n=10k, 100 runs")    
    #row = [0.700202,0.102843,0.0479883,0.027656,0.017853,0.0126456,0.0095235,0.0072096,0.0058977,0.004749,0.004015,0.0034608,0.0028743,0.0025564,0.002325,0.001976,0.0018037,0.0015714,0.0014193,0.001276,0.00120697,0.0010692,0.00105143,0.000937732,0.000855,0.000817526,0.000696477,0.000756596,0.000728372,0.000718681,0.000684884,0.000655238,0.000651538,0.000653158,0.000585,0.000606575,0.000598529,0.000674194,0.000625238,0.000586207]
    #plt.plot(range(1, cut_off), row, marker='x', fillstyle='none', linestyle='none', label="sim, n=100k, 100 runs")
    #row = [0.700029,0.102841,0.0477795,0.0274372,0.018099,0.0127746,0.0095753,0.007332,0.0059238,0.00473,0.0040249,0.0032892,0.0028587,0.0025088,0.002265,0.002056,0.0017748,0.0015228,0.0014193,0.001274,0.0011508,0.0011088,0.0009982,0.0007968,0.000845,0.0008242,0.0007506,0.0007,0.0006032,0.000513,0.0006014,0.000576,0.0005511,0.0004522,0.00049,0.0004392,0.0004551,0.0004142,0.0004095,0.000288]
    #plt.plot(range(1, cut_off), row, marker='x', fillstyle='none', linestyle='none', label="sim, n=1M, 10 runs")   
    
    row = [0.700255,0.103169,0.0477633,0.0274832,0.0179515,0.0127272,0.0094591,0.0073792,0.0058365,0.004754,0.0039413,0.0034224,0.0029029,0.002408,0.0022065,0.0019712,0.0018105,0.001692,0.0012958,0.001314,0.0011151,0.0010736,0.000973434,0.00098449,0.00093883,0.00086125,0.000827234,0.000766957,0.000742527,0.000654878,0.000685647,0.000564938,0.000619259,0.000626098,0.000652273,0.000596712,0.000596418,0.000584615,0.000635323,0.000608696,0.000656,0.0006216,0.000516,0.0005456,0.0005625,0.00062898,0.000661837,0.000585,0.000601364,0.0006,0.000756774,0.000668571,0.000612812,0.000671351,0.000701724,0.000708235,0.000692143,0.0006496,0.000624706,0.0007125,0.000691333,0.000717895,0.00075,0.000822857,0.000680952,0.00066,0.000747308,0.00068,0.000784091,0.000827273,0.000742273,0.000754286,0.00073,0.000781111,0.000865385,0.000818462,0.000829231,0.00078,0.000877778,0.0008,0.00081,0.00082,0.00083,0.00084,0.00085,0.00086,0.0009425,0.00088,0.000979,0.0009,0.00091,0.00092,0.00093,0.00107429,0.00095,0.00096,0.00097,0.001078,0.00099,0.001,0.00101,0.00102,0.00103,0.00104,0.00105,0.00106,0.00107,0.00108,0.00109,0.0011,0.00111,0.00112,0.00113,0.00114,0.00131429,0.00116,0.001404,0.001475,0.00119,0.0012,0.00121,0.00122,0.00123,0.00124,0.00125,0.00126,0.001524,0.00128,0.00129,0.0013,0.00131,0.00132,0.00133,0.00134,0.00135,0.00136,0.00137,0.00138,0.00139,0.0014,0.00141,0,0.00143,0.00144,0.00145,0.00146,0.00147,0.00148,0.00149,0.0015,0.00151,0.00152,0,0.00154,0.00155,0,0.00157,0.00158,0.00159,0.0016,0.00161,0.00162,0.00163,0.00164,0.00165,0,0.00167,0.00168,0,0.0017,0.00171,0.00172,0.00173,0.00174,0.00175,0.00176,0.00177,0.00178,0.00179,0.0018,0.00181,0,0.00183,0.00184,0.00185,0.00186,0.00187,0.00188,0.00189,0.0019,0.00191,0.00192,0,0.00194,0.00195,0.00196,0.00197,0.00198,0.00199,0.002,0.00201,0.00202,0,0.00204,0.00205,0.00206,0.00207,0,0.00209,0.0021,0,0.00212,0.00213,0.00214,0,0,0,0.00218,0.00219,0.0022,0.003315,0.00222,0,0,0,0,0.00227,0.00228,0,0,0,0.00232,0,0,0.00235,0,0,0.00238,0,0.0024,0,0,0.00243,0,0,0,0.00247,0,0,0.0025,0,0,0,0.00254,0,0.00256,0.00257,0,0,0,0.00261,0,0.00263,0,0.00265,0,0,0.00268,0,0,0,0,0.00273,0,0,0,0.00277,0.00278,0,0,0,0,0.00283,0,0.00285,0.00286,0,0.00288,0,0,0,0.00292,0,0,0,0.00296,0,0,0.00299,0.003,0,0,0,0,0,0,0.00307,0,0,0,0.00311,0,0,0,0,0,0,0,0,0,0,0.00322,0,0,0,0.00326,0,0,0.00329,0.0033,0,0.00332,0,0.00334,0,0,0.00337,0,0,0.0034,0,0.00342,0.00343,0,0,0,0,0,0,0,0,0,0,0.00354,0,0.00356,0,0,0.00359,0,0,0,0,0,0,0,0.00367,0,0,0,0,0,0,0,0,0,0,0.00378,0,0,0.00381,0,0,0,0,0.00386,0.00387,0,0.00389,0,0,0,0,0,0,0.00396,0,0,0.00399,0,0,0,0,0,0,0.00406,0,0,0,0,0,0,0.00413,0,0,0.00416,0,0.00418,0,0,0,0,0.00423,0,0,0,0,0,0,0,0,0,0,0,0,0.00436,0,0,0.00439,0,0.00441,0.00442,0.00443,0,0,0,0.00447,0,0,0,0,0,0,0,0.00455,0,0,0,0,0.0046,0,0,0,0,0.00465,0,0.00467,0,0,0,0.00471,0.00472,0,0.00474,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00499,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00513,0,0,0,0,0,0.00519,0,0,0,0,0,0,0.00526,0,0,0,0,0,0.00532,0,0.00534,0.00535,0.00536,0,0,0,0,0,0.00542,0,0.00544,0.00545,0,0,0,0,0,0,0,0,0,0,0.00556,0,0,0,0,0.00561,0.00562,0.00563,0.00564,0,0,0,0.00568,0,0,0,0.00572,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00594,0.00595,0,0,0,0,0,0,0.00602,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00628,0,0.0063,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00645,0.00646,0,0,0,0,0,0,0,0,0,0,0.00657,0,0,0.0066,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00694,0.00695,0,0,0,0.00699,0,0,0.00702,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00721,0,0,0.00724,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00773,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00798,0,0,0.00801,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00824,0,0,0,0,0,0.0083,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00862,0,0,0,0,0,0,0,0,0,0,0,0,0.00875,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00914,0,0,0,0,0,0.0092,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00939,0,0.00941,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00957,0,0,0,0,0.00962,0,0,0,0,0,0,0,0,0.00971,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00986,0,0,0.00989,0,0,0,0,0,0,0,0,0,0,0]
    plt.plot(range(1, cut_off), row[:cut_off-1], marker='x', fillstyle='none', linestyle='none', label="sim, n=10M, 1 runs")   
    #for phi in ax:
        #ps = [test(phi, s, n, p) for s in range(1, cut_off)]
        #x.append(sum(ps[i-1]*i for i in range(1, cut_off))/sum(i for i in ps))
        #plt.plot(range(1, cut_off),  ps, marker='o', fillstyle='none', label="analytical solution")
        #print("sum", sum(ps))
    #plt.plot(ax, x, marker='o', fillstyle='none')
    plt.plot(range(1, cut_off), [factorial(3*s-3, True)/(factorial(s-1, True)*factorial(2*s-1, True)) * 0.3**(s-1)*(1-0.3)**(2*s-1) for s in range(1, cut_off)], marker='o', fillstyle='none', label='analytical sol.')
    #plt.xlim([0, 10])
    plt.yscale("log")
    """
    #x = np.arange(0, 1+1/len(row_1), 1/(len(row_1)-1))
    truth = []
    for phi in x2:
        print(phi)
        if(phi == 0):
            truth.append(0.0)
            continue
        temp = [test(phi, i, n, p) for i in range(1, cut_off)]
        truth.append(sum([v*i for i, v in enumerate(temp, start=1)])/sum(temp))
    truth = [test(phi, 3, n, p) for phi in x]
    plt.plot(x, row_3_3, marker='x', fillstyle='none', linestyle='none', label="sim, n=1k, 100 runs")

    plt.plot(x[5:], truth[5:], marker='o', fillstyle='none', label="analytical solution")
    """
    plt.legend()
    plt.grid(True)
    #plt.title("Small component distribution - πs\nER network - <k> = 1.5")
    plt.title("Small component distribution - πs\nExp. degree dist. - alpha = 0.3")
    plt.xlabel("Component Size s")
    #plt.xlabel("occupation prob.")
    plt.ylabel("πs")# - s=3")
    plt.show()

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

        if exp_params['percolation_type'] == 'f':
            featurebondperc(exp_n, exp_params, row)
            return

        if exp_params['percolation_type'] == 'c':
            corrfeaturebondperc(exp_n, exp_params, row)
            return

        if exp_params['percolation_type'] == 's':
            smallcomponent(exp_n, exp_params, row)
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
    #small_comp_dist()