import yaml
import csv
import multiprocess as mp
import numpy as np
from math import asin, sin
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import binom, geom, poisson
from scipy.misc import derivative
from scipy.special import factorial
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

    def featureDist(self, k, t):
        featdist = [poisson.pmf(f, 8) for f in self.features_bins]

        def f(x, t):
            A = 10
            k = 10
            phi = asin((x-k)/A)
            return int(A*sin(2*3.1415*t/30.0 + phi)+k)

        return sum([val for i, val in enumerate(featdist) if f(i, t)==k])

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = np.zeros_like(self.exp_data)
        def func(t):
            result = []
            for F0 in self.features_bins:
                psi = sum([self.featureDist(f, t) for f in range(0, F0)])
                def f(u):
                    return u - 1 + psi - psi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for F0={}".format(F0))
                result.append(1-self.g0(sol.x, degdist, lower_limit, upper_limit))
            return t, result


        with alive_bar(len(self.time_bins), theme='smooth') as bar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val in pool.imap_unordered(func, self.time_bins):
                    self.sol_data[idx] = val
                    bar()

    def getPlot(self, description):
        
        plt.imshow(self.exp_data, cmap='viridis')
        plt.colorbar(label="Size of giant cluster")
        plt.ylim(-0.5, self.time_bins.size-0.5)
        plt.title("Temporal features edge percolation \n"+description)
        plt.xlabel("Maximum Feature F0")
        plt.ylabel("Time t")
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
        super().__init__(net_size, data_path)

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        phi = 1
        with alive_bar(len(self.bins), theme='smooth') as bar:
            for s in self.bins:
                def f(u):
                    return u - 1 + phi - phi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for s={}".format(s))

                S = phi*(1-self.g0(sol.x))

                def f2(z):
                    self.g1(z, excdegdist, lower_limit, upper_limit)**s

                if(s==1):
                    self.sol_data.append(0)
                else:
                    mean = 3
                    self.sol_data.append(phi*derivative(f2, 1-phi, n=s-2, dx=1e-2, order=21)*mean*phi**(s-1)/factorial(s-1))
                bar()
    
    def getPlot(self, description):
        plt.plot(self.bins, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins, self.sol_data, label="Analytical solution")
        #plt.xlim((-0.1, 1.1))
        #plt.ylim((-0.1, 1.1))
        plt.yscale("log")
        plt.title("Small component distribution - πs\n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Component Size s")
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
               'g': lambda k: geom.pmf(k+1, param1),
               'f': lambda k: 1 if k==int(param1) else 0}[net_type]

    degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])

    def excdegdist(k):
        return degdist(k+1)*(k+1)/degdistmean
    
    #plotdistribution(lambda k: degdist(k)-excdegdist(k), lower_limit, upper_limit).show()

    plotter.computeAnalitycalSolution(degdist, excdegdist, lower_limit, upper_limit)
    plt = plotter.getPlot(subtitle)
    plt.show()

if __name__ == "__main__":
    main()

    
    

    