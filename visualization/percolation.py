import yaml
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import binom
from alive_progress import alive_bar

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

    def g0(self, z, degdist, upper_limit):
        return sum([degdist(k)*z**k for k in range(0, upper_limit)])

    def g1(self, z, excdegdist, upper_limit):
        return sum([excdegdist(k+1)*z**k for k in range(0, upper_limit)])

    def computeAnalitycalSolution(self, degdist, excdegdist, upper_limit=50):
        self.sol_data = []
        with alive_bar(len(self.bins), theme='smooth') as bar:
            for phi in self.bins:
                def f(u):
                    return u - 1 + phi - phi * self.g1(u, excdegdist, upper_limit)

                sol = optimize.root(f, 0.5)
                if not sol.success:
                    raise AssertionError("Solution not found for phi={}".format(phi))
                self.sol_data.append(phi*(1-self.g0(sol.x, degdist, upper_limit)))
                bar()

    def getPlot(self, description):
        plt.plot(self.bins, self.exp_data, marker='o', fillstyle='none', linestyle='none', label='Experimental results')
        plt.plot(self.bins, self.sol_data, label="Analytical solution")
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.title("Average size of the largest cluster as a function of φ \n"+description)
        plt.legend(loc='upper left')
        plt.xlabel("Occupation probability φ")
        plt.ylabel("Size of giant cluster S(φ)")
        plt.grid(True)
        return plt


def main():
    upper_limit = 50

    with open("./experiments/test.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    if params['percolation_type'] == 'n':
        plotter = NodeUniformRemoval(int(params['network_size']), "./results/raw/percolation_result.csv")

    degdist = {'b': lambda k: binom.pmf(k, int(params['network_size']), float(params['param1']))}[params['network_type']]
    degdistmean = sum([k*degdist(k) for k in range(0, upper_limit)])
    def excdegdist(k):
        return degdist(k+1)*(k+1)/degdistmean
    
    plotter.computeAnalitycalSolution(degdist, excdegdist, upper_limit)
    plt = plotter.getPlot("TEST")
    plt.show()

if __name__ == "__main__":
    main()

    
    

    