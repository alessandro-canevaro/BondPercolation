import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.stats import binom, geom, poisson
from alive_progress import alive_bar
from warnings import filterwarnings

filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
filterwarnings("ignore", category=DeprecationWarning) 

data_dir = "./data/biogrid/"
exp_data_path = "./results/raw/percolation_result.csv"
num_files = 1#5
dx = 0.1
min_f, max_f = -100, 100
max_feat = int(2*(max_f/dx)+1)
max_deg = 200
shift, scale = int(min_f/dx), int(1/dx) #[0, 0, 0, 0, 0, 0, 0], 1#[0, 14, -246, 0, +1, -1], 10

def from_dict_to_func(dist_dict, max_val, nodes=-1):
    dist_list = [dist_dict[i] if i in dist_dict else 0 for i in range(max_val+1)]
    #print("before", dist_list, sum(dist_list))
    if nodes > 0:
        dist_list[0] = nodes-sum(dist_list)
    sum_list = sum(dist_list)
    #print(dist_list, sum_list)
    if sum_list==0:
        return lambda k: 0
    dist_list = [val/sum_list for val in dist_list]
    return lambda k: dist_list[k] if 0 <= k <= max_val else 0

class UncorrelatedFeatureEdgePercolation:
    
    def __init__(self, net_size, exp_data) -> None:
        self.nodes = net_size
        self.exp_data = exp_data
        self.exp_data = [float(i)/self.nodes for i in self.exp_data]
        self.bins = np.arange(0, len(self.exp_data))

    def g0(self, z, degdist, upper_limit):
        return sum([degdist(k)*z**k for k in range(0, upper_limit)])

    def g1(self, z, excdegdist, upper_limit):
        return sum([excdegdist(k)*z**k for k in range(0, upper_limit)])

    def computeAnalitycalSolution(self, featdist, degdist, excdegdist, upper_limit, pgr_bar):
        self.sol_data = []
        for F0 in self.bins:
            psi = sum([featdist(f) for f in range(0, F0)])
            def f(u):
                return u - 1 + psi - psi * self.g1(u, excdegdist, upper_limit)

            sol = optimize.root(f, 0, method='lm')
            if not sol.success:
                sol.x=0
                print('raise AssertionError("Solution not found for F0={}".format(F0))')
            if sol.x > 1.0:
                sol.x = 1.0
            self.sol_data.append(1-self.g0(sol.x, degdist, upper_limit))
            pgr_bar()
    
    def getPlot(self, plt_ax, description, shift=0, scale=1):
        plt_ax.plot((self.bins+shift)/scale, self.exp_data, label='Experimental results')#, linestyle='none', marker='o', fillstyle='none')
        plt_ax.plot((self.bins+shift)/scale, self.sol_data, label="Analytical solution")#, marker='o', fillstyle='none')
        print(shift, scale, len(self.exp_data))
        if(not self.exp_data):
            return
        #plt_ax.set_xlim(((-0.5+shift)/scale, (len(self.bins)-0.5+shift)/scale))
        plt_ax.set_xlim((-10, 10))
        plt_ax.set_ylim((-0.1, 1.1))
        plt_ax.set_title(description)#"Uncorrelated features edge percolation \n"+
        #plt_ax.legend(loc='upper left')
        #plt_ax.set_xlabel("Maximum Feature F0")
        #plt_ax.set_ylabel("Size of giant cluster")
        plt_ax.grid(True)
        #return plt

def plotdistribution(plt_ax, dist, lower_limit, upper_limit, shift=0, scale=1):
    x = np.arange(lower_limit, upper_limit)
    y = [dist(k) for k in x]
    plt_ax.bar((x+shift/scale), y)
    #plt_ax.title("Probability distribution")
    #plt_ax.set_xlabel("degree")
    #plt_ax.set_ylabel("density")
    plt_ax.grid(True)
    #return plt

def main():
    names = ['direct_interaction', 'positive_genetic_interaction', 'negative_genetic_interaction', 'synthetic_growth_defect', 'colocalization', 'physical_association']
    with open(exp_data_path) as file:
        exp_data = next(csv.reader(file))

    num_rows, num_cols = 1, 1#2, 3

    fig_perc, ax_perc = plt.subplots(nrows=num_rows, ncols=num_cols, constrained_layout=True, sharex=True, sharey=True)
    #fig_feat, ax_feat = plt.subplots(nrows=num_rows, ncols=num_cols, constrained_layout=True, sharex=True, sharey=True)

    with alive_bar(num_files*max_feat, theme='smooth') as bar:
        for i, f in enumerate(sorted(glob.glob(data_dir+"aggregation.txt"))):#enumerate(sorted(glob.glob(data_dir+"*.txt"))):
            #print("Current File Being Processed is: " + f)
            df = pd.read_csv(f, sep=" ", names=["from", "to", "feature"])
            #print(df)

            nodes = max(df[["from", "to"]].max().to_list()) + 1 

            #compute feature dist.
            feat_dist_dict = df["feature"].value_counts().to_dict()
            featdist = from_dict_to_func(feat_dist_dict, max_feat)

            df_from_to = df["from"].append(df["to"], ignore_index = True)
            deg_dist_dict = df_from_to.value_counts().value_counts().to_dict()
            degdist = from_dict_to_func(deg_dist_dict, max_deg, nodes=nodes)

            #plotdistribution(degdist, 0, max_deg).show()

            degdistmean = sum([k*degdist(k) for k in range(0, max_deg)])
            #print("deg. dist. mean:", degdistmean, "- nodes:", nodes)

            def excdegdist(k):
                return degdist(k+1)*(k+1)/degdistmean

            plotter = UncorrelatedFeatureEdgePercolation(nodes, exp_data[i*max_feat:(i+1)*max_feat])#[feat_range[i][0]:feat_range[i][1]])
            plotter.computeAnalitycalSolution(featdist, degdist, excdegdist, max_deg, bar)
            plotter.getPlot(ax_perc, "Aggregation", shift, scale)
            #plotter.getPlot(ax_perc[i//num_cols, i%(1+num_rows)], names[i], shift, scale)
            #plotdistribution(ax_feat[i//num_cols, i%(1+num_rows)], featdist, 0, max_feat, shift, scale)#-10)

    lines, labels = ax_perc.get_legend_handles_labels()
    #lines, labels = ax_perc[0, 0].get_legend_handles_labels()
    fig_perc.legend(lines, labels, loc='upper right')

    fig_perc.supxlabel("Maximum Feature F0")
    fig_perc.supylabel("Size of giant cluster")
    fig_perc.suptitle("Uncorrelated features edge percolation - Homo sapiens dataset\n"+"Feature range: [-100, 100], dx=0.1")
    """
    fig_feat.supxlabel("Feature value")
    fig_feat.supylabel("Probability")
    fig_feat.suptitle("Feature distribution - 25 windows \n"+"Bitcoin ALPHA trust weighted signed network")
    """
    plt.show()

if __name__ == "__main__":
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))
    main()
