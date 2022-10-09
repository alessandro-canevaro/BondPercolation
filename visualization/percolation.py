from logging import critical
import yaml
import csv
import multiprocess as mp
import numpy as np
from math import asin, sin
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import binom, geom, poisson
from scipy.misc import derivative
from scipy.special import factorial#, gamma, gammaincc
from mpmath import gamma, gammainc
from alive_progress import alive_bar
from warnings import filterwarnings

filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
filterwarnings("ignore", category=DeprecationWarning) 

class AnalitycalSolution:

    def __init__(self, net_size, data_path) -> None:
        self.nodes = net_size

        with open(data_path) as file:
            self.exp_data = next(csv.reader(file))

        self.bins = np.arange(0, 1+1/len(self.exp_data), 1/(len(self.exp_data)-1))

        self.exp_data = [float(i)/self.nodes for i in self.exp_data]

    def getPlot(self):
        return plt.plot(self.bins, self.exp_data)

class NodeUniformRemoval(AnalitycalSolution):

    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)

    def g0(self, z, degdist, lower_limit, upper_limit):
        return sum([degdist(k)*z**k for k in range(lower_limit, upper_limit)])

    def g1(self, z, excdegdist, lower_limit, upper_limit):
        return sum([excdegdist(k)*z**k for k in range(lower_limit, upper_limit)])

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        with alive_bar(len(self.bins), theme='smooth') as bar:
            for phi in self.bins:
                def f(u):
                    return u - 1 + phi - phi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for phi={}".format(phi))
                self.sol_data.append(phi*(1-self.g0(sol.x, degdist, lower_limit, upper_limit)))
                bar()

    def getPlot(self, description):
        plt.plot(self.bins, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins, self.sol_data, label="Analytical solution")
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.title("Uniform random removal node percolation \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Occupation probability φ")
        plt.ylabel("Size of giant cluster S(φ)")
        plt.grid(True)
        return plt

class NodeTargetedAttack(AnalitycalSolution):

    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)
        self.bins = np.arange(0, len(self.exp_data))

    def f0(self, z, degdist, lower_limit, k0):
        return sum([degdist(k)*z**k for k in range(lower_limit, k0)])

    def f1(self, z, excdegdist, lower_limit, k0):
        return sum([excdegdist(k)*z**k for k in range(lower_limit, k0-1)])

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        with alive_bar(len(self.bins), theme='smooth') as bar:
            for k0 in self.bins:
                def f(u):
                    return u - 1 + self.f1(1, excdegdist, lower_limit, k0) - self.f1(u, excdegdist, lower_limit, k0)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for k0={}".format(k0))
                self.sol_data.append(self.f0(1, degdist, lower_limit, k0) - self.f0(sol.x, degdist, lower_limit, k0))
                bar()

    def getPlot(self, description):
        plt.plot(self.bins, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins, self.sol_data, marker='o', fillstyle='none', label="Analytical solution")
        plt.xlim((-0.5, len(self.bins)-0.5))
        plt.ylim((-0.1, 1.1))
        plt.title("Targeted attack node percolation \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Maximum degree k0")
        plt.ylabel("Size of giant cluster")
        plt.grid(True)
        return plt

class EdgeUniformRemoval(NodeUniformRemoval):

    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        super().computeAnalitycalSolution(degdist, excdegdist, lower_limit, upper_limit)
        self.sol_data = [0]+[data/phi for data, phi in zip(self.sol_data, self.bins) if phi>0]

    def getPlot(self, description):
        plt = super().getPlot(description)
        plt.title("Uniform random removal edge percolation \n"+description)
        return plt

class UncorrelatedFeatureEdgePercolation(NodeUniformRemoval):
    
    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)
        self.bins = np.arange(0, len(self.exp_data))

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        with alive_bar(len(self.bins), theme='smooth') as bar:
            for F0 in self.bins:
                psi = sum([poisson.pmf(f, 8) for f in range(0, F0)])
                def f(u):
                    return u - 1 + psi - psi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for F0={}".format(F0))
                self.sol_data.append(1-self.g0(sol.x, degdist, lower_limit, upper_limit))
                bar()
    
    def getPlot(self, description):
        plt.plot(self.bins, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins, self.sol_data, marker='o', fillstyle='none', label="Analytical solution")
        plt.xlim((-0.5, len(self.bins)-0.5))
        plt.ylim((-0.1, 1.1))
        plt.title("Uncorrelated features edge percolation \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Maximum Feature F0")
        plt.ylabel("Size of giant cluster")
        plt.grid(True)
        return plt

class CorrelatedFeatureEdgePercolation(UncorrelatedFeatureEdgePercolation):

    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):

        upper_limit = 10

        def pfkk(f, m):
            return poisson.pmf(f, m)

        def psi(u, k, F0):
            return sum([sum([excdegdist(m-1) * pfkk(f, (m+k)) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])

        def func(F0):
            def vecfunc(u):
                result = np.zeros_like(u)
                for k in range(1, upper_limit):
                    result[k-1] = (1-psi(u, k, F0))**(k-1)
                return u-result

            sol = optimize.root(vecfunc, np.zeros((upper_limit-1, 1))+0.0, method='lm')
            if not sol.success:
                print("ERROR")

            W = np.zeros((upper_limit, 1))
            W[0] = 1
            for k in range(1, upper_limit):
                W[k] = (1-psi(sol.x, k, F0))**k

            S = sum([degdist(k)*(1-W[k]) for k in range(0, upper_limit)])
            return (F0, S[0])

        self.sol_data = [0]*len(self.bins)
        with alive_bar(len(self.bins), theme='smooth') as bar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val in pool.imap_unordered(func, self.bins):
                    self.sol_data[idx] = val
                    bar()

    def computeAnalitycalSolutionUnrolled(self, degdist, excdegdist, lower_limit, upper_limit):

        degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])

        upper_limit = 10

        def pfkk(f, m):
            return poisson.pmf(f, m)

        def psi(u, k, F0):
            
            a = np.exp(-degdistmean)/gamma(F0+1) * sum([u[m-1]*degdistmean**m / factorial(m) * gammainc(F0+1, k+m+1) for m in range(1, upper_limit)])
            #b = sum([sum([excdegdist(m-1) * pfkk(f, (m+k)) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])
            #print(a, b)
            return a

        def func(F0):
            def vecfunc(u):
                result = np.zeros_like(u)
                for k in range(1, upper_limit):
                    result[k-1] = (1-psi(u, k, F0))**(k-1)
                return u-result

            sol = optimize.root(vecfunc, np.zeros((upper_limit-1, 1))+0.0, method='lm')
            if not sol.success:
                print("ERROR")

            W = np.zeros((upper_limit, 1))
            W[0] = 1
            for k in range(1, upper_limit):
                W[k] = (1-psi(sol.x, k, F0))**k

            S = sum([degdist(k)*(1-W[k]) for k in range(0, upper_limit)])
            return (F0, S[0])

        self.sol_data = [0]*len(self.bins)
        with alive_bar(len(self.bins), theme='smooth') as bar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val in pool.imap_unordered(func, self.bins):
                    #quit()
                    self.sol_data[idx] = val
                    bar()


    def getPlot(self, description):
        plt = super().getPlot(description)
        plt.title("Correlated features edge percolation \n"+description)
        return plt

class TemporalFeatureEdgePercolation(UncorrelatedFeatureEdgePercolation):

    def __init__(self, net_size, data_path) -> None:
        self.nodes = net_size
        self.features_bins = np.arange(0, 20)
        self.time_bins = np.arange(0, 30)

        with open(data_path) as file:
            self.exp_data = next(csv.reader(file))

        self.exp_data = [float(i)/self.nodes for i in self.exp_data]
        self.exp_data = np.array(self.exp_data).reshape(self.time_bins.size, self.features_bins.size)

        self.critical_points = [0]*self.time_bins

    def featureDist(self, k, t):
        featdist = [poisson.pmf(f, 8) for f in self.features_bins]

        def f(x, t):
            A = 10
            k = 10
            phi = asin((x-k)/A)
            return int(A*sin(2*3.1415*t/30.0 + phi)+k)

        return sum([val for i, val in enumerate(featdist) if f(i, t)==k])

    def g1p(self, z, excdegdist, lower_limit, upper_limit):
        return sum([excdegdist(k-1)*(k-1)*z**(k-2) for k in range(2, upper_limit)])

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = np.zeros_like(self.exp_data)

        def func(t):
            result = []
            critical_point = 0
            for F0 in self.features_bins:
                psi = sum([self.featureDist(f, t) for f in range(0, F0)])
                def f(u):
                    return u - 1 + psi - psi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for F0={}".format(F0))

                if self.g1p(sol.x, excdegdist, lower_limit, upper_limit) > 1:
                    critical_point = F0-1 if F0 > 0 else 0

                result.append(1-self.g0(sol.x, degdist, lower_limit, upper_limit))
            return t, result, critical_point


        with alive_bar(len(self.time_bins), theme='smooth') as bar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val, point in pool.imap_unordered(func, self.time_bins):
                    self.sol_data[idx] = val
                    self.critical_points[idx] = point
                    bar()
        #print(self.critical_points)

    def getPlot(self, description):
        plt.imshow(self.exp_data, cmap='viridis')
        plt.scatter(self.critical_points, self.time_bins, marker='4', color='red', label="Critical point")
        plt.colorbar(label="Size of giant cluster")
        plt.ylim(-0.5, self.time_bins.size-0.5)
        plt.title("Temporal features edge percolation \n"+description)
        plt.xlabel("Maximum Feature F0")
        plt.ylabel("Time t")
        plt.legend()
        plt.show()
        plt.clf()

        for t in [0, 3, 19]:
            plt.plot(self.features_bins, self.exp_data[t, :], marker='o', fillstyle='none', linestyle='none')#, label='Experimental results {}'.format(t))
            plt.plot(self.features_bins, self.sol_data[t, :], marker='o', fillstyle='none', label='Analytical sol. - t={}'.format(t))
        plt.xlim((-0.5, len(self.features_bins)-0.5))
        plt.ylim((-0.1, 1.1))
        plt.title("Temporal features edge percolation \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Maximum Feature F0")
        plt.ylabel("Size of giant cluster")
        plt.grid(True)
        plt.show()
        plt.clf()
        
        for f in [3, 10, 17]:
            plt.plot(self.time_bins, self.exp_data[:, f], marker='o', fillstyle='none', linestyle='none')#, label='Experimental results {}'.format(f))
            plt.plot(self.time_bins, self.sol_data[:, f], marker='o', fillstyle='none', label='Analytical sol. - f={}'.format(f))
        plt.xlim((-0.5, len(self.time_bins)-0.5))
        plt.ylim((-0.1, 1.1))
        plt.title("Temporal features edge percolation \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Time t")
        plt.ylabel("Size of giant cluster")
        plt.grid(True)
        
        return plt

class SmallComponents(NodeUniformRemoval):

    def __init__(self, net_size, data_path) -> None:
        self.nodes = net_size

        with open(data_path) as file:
            self.exp_data = next(csv.reader(file))

        self.exp_data = [float(i) for i in self.exp_data[:99]]
        self.bins_phi = np.arange(0, 1+1/len(self.exp_data), 1/(len(self.exp_data)-1))
        self.bins_s = np.arange(1, len(self.exp_data)+1)

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = np.zeros_like(self.exp_data)
        s = 3

        def func(idx):
            phi = self.bins_phi[idx]

            def f(u):
                return u - 1 + phi - phi * self.g1(u, excdegdist, lower_limit, upper_limit)

            sol = optimize.root(f, 0.6)
            if not sol.success:
                print("Solution not found for phi={}".format(phi))
                return 0, idx
                raise AssertionError("Solution not found for phi={}".format(phi))

            def f2(z):
                return self.g1(z, excdegdist, lower_limit, upper_limit)**s
        
            if(s==1):
                return phi*sum([degdist(k)*(1-phi)**k for k in range(lower_limit, upper_limit)]), idx
            else:
                mean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
                return phi*derivative(f2, 1-phi, n=s-2, dx=1e-2, order=51)*mean*phi**(s-1)/factorial(s-1), idx

        with alive_bar(len(self.bins_phi), theme='smooth') as bar:
            with mp.Pool(mp.cpu_count()) as pool:
                for val, idx in pool.imap_unordered(func, list(range(len(self.bins_phi)))):
                    self.sol_data[idx] = val
                    bar()

    def computeAnalitycalSolution2(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        phi = 1
        a = 0.5
        with alive_bar(len(self.bins_s), theme='smooth') as bar:
            for s in self.bins_s:
                true_sol = factorial(3*s-3, True)/(factorial(s-1, True)*factorial(2*s-1, True)) * a**(s-1)*(1-a)**(2*s-1) #true analytical solution
                #print(s, true_sol)
                self.sol_data.append(true_sol) 
                bar()
                continue

                def f(u):
                    return u - 1 + phi - phi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0.5)
                if not sol.success:
                    raise AssertionError("Solution not found for s={}".format(s))

                def f2(z):
                    return self.g1(z, excdegdist, lower_limit, upper_limit)**s
            
                if(s<=1):
                    self.sol_data.append(degdist(0))
                else:
                    mean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
                    self.sol_data.append(phi*derivative(f2, 1-phi, n=s-2, dx=1e-2, order=151)*mean*phi**(s-1)/factorial(s-1))
                
                bar()

        print("analytical sol. sum:", sum(self.sol_data), "exp. data sum:", sum(self.exp_data))
    
    def getPlot(self, description):
        """
        plt.plot(self.bins_s, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins_s, self.sol_data, marker='o', fillstyle='none', label="Analytical solution")
        data_1M = [0.709711,0.103964,0.0481009,0.028009,0.0182806,0.012728,0.009467,0.00742367,0.00608137,0.00459927,0.00392255,0.00340387,0.00316073,0.00260963,0.00235535,0.00199369,0.0018083,0.00154997,0.00155909,0.00127645,0.00125518,0.000757765,0.00102521,0.000972533,0.00116501,0.000711165,0.000957337,0.000595676,0.000910737,0.000577441,0.000659499,0.000324178,0.000468032,0.000516658,0.000354569,0.00040117,0.00037483,0.000423457,0.000395092,0.000243133,0.000373817,0.000340387,0.000174246,0.000312021,9.1175e-05,0.000512606,0.000190454,0.000243133,0.000198559,0.000253264,0.000103332,0.000263394,0.000214768,0.00032823,0,5.67311e-05,0.000173232,0.000293786,0.000239081,0.00018235,6.17964e-05,0,0.000127645,6.48355e-05,0.000131697,0.000133723,6.78747e-05,0.000482214,0,0.000354569,0.000215781,0.00029176,0,7.49661e-05,0.000151958,7.69922e-05,7.80053e-05,7.90183e-05,8.00314e-05,0.000162089,8.20575e-05,8.30705e-05,8.40836e-05,0,0.000172219,0,8.81358e-05,0,0,9.1175e-05,0.000184376,0.000279603,0.000188428,0,9.62402e-05,9.72533e-05,9.82664e-05,9.92794e-05,0.000200585]
        data_10k = [0.708346,0.0997471,0.0479514,0.0307537,0.0161861,0.0151745,0.00849772,0.00566515,0.0100152,0.00505817,0.0011128,0.00364188,0.00131512,0,0.00151745,0.00323723,0.00171978,0.00364188,0.0019221,0.00202327,0,0,0.00232676,0,0,0,0,0,0,0,0,0,0.00333839,0.00343955,0,0,0,0,0.00394537,0,0.0041477,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00657562,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.00870005,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #plt.plot(self.bins_s, data_10k[:99], marker='o', fillstyle='none', linestyle='none', label='single run')
        #plt.ylim((1e-5, 1e-0))
        plt.yscale("log")
        plt.title("Small components distribution - πs\n"+description)
        plt.legend(loc='upper right')
        plt.xlabel("Component Size s")
        plt.ylabel("πs")
        plt.grid(True)
        """

        plt.plot(self.bins_phi, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins_phi, self.sol_data, marker='o', fillstyle='none', label="Analytical solution")
        plt.title("Small component distribution - s=3\n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Occupation probability φ")
        plt.ylabel("πs")
        plt.grid(True)
        return plt

def plotdistribution(dist, lower_limit, upper_limit):
    x = np.arange(lower_limit, upper_limit)
    y = [dist(k) for k in x]
    plt.bar(x, y)   # plot circles...
    plt.title("Probability distribution")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.grid(True)
    return plt

def main():
    lower_limit = 0
    upper_limit = 50
    C = 1
    exp_data_path = "./results/raw/percolation_result.csv"

    with open("./experiments/test.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        net_size = int(params['network_size'])
        net_type = params['network_type']
        perc_type = params['percolation_type']
        param1 = float(params['param1'])

    plotter = {'n': NodeUniformRemoval, #node percolation uniform random removal
               'a': NodeTargetedAttack, #targeted attack node percolation
               'l': EdgeUniformRemoval, #edge percolation uniform random removal
               'f': UncorrelatedFeatureEdgePercolation, #edge percolation uncorrelated features
               'c': CorrelatedFeatureEdgePercolation, #edge percolation correlated features
               't': TemporalFeatureEdgePercolation, #edge percolation temporal features
               's': SmallComponents, #small components node percolation
               '': None}[perc_type](net_size, exp_data_path)

    if net_type == 'p':
        upper_limit = int(net_size**0.5)+1
        C = sum([ks**(-param1) for ks in range(lower_limit, upper_limit) if ks>=2])

    subtitle = {'b': "ER network ⟨k⟩ = {:.1f}; n={}; runs={}".format(net_size*param1, net_size, params['runs']),
                'p': "SF network α = {:.1f}; n={}; runs={}".format(param1, net_size, params['runs']),
                'g': "Geometric degree dist. network a = {:.1f}; n={}; runs={}".format(param1, net_size, params['runs']),
                'f': "Fixed degree network k = {:.1f}; n={}; runs={}".format(int(param1), net_size, params['runs'])}[net_type]

    degdist = {'b': lambda k: binom.pmf(k, net_size, param1),
               'p': lambda k: (k**(-param1)/C) if k>=2 else 0,
               'g': lambda k: geom.pmf(k+1, 1-param1),
               'f': lambda k: 1 if k==int(param1) else 0}[net_type]

    degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
    print("deg. dist. mean:", degdistmean)

    def excdegdist(k):
        return degdist(k+1)*(k+1)/degdistmean
    
    #plotdistribution(lambda k: degdist(k)-excdegdist(k), lower_limit, upper_limit).show()

    plotter.computeAnalitycalSolution(degdist, excdegdist, lower_limit, upper_limit)
    plt = plotter.getPlot(subtitle)
    #plt.savefig("./results/figures/node_perc_giant_cluster_exp_{}.png".format(0))
    plt.show()

if __name__ == "__main__":
    main()

    
    

    
