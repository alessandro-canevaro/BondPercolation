import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocess as mp
from scipy import optimize
from scipy.stats import binom, geom, poisson
from scipy.special import factorial, gammaincc
from alive_progress import alive_bar
from tqdm import tqdm
from warnings import filterwarnings
import warnings
warnings.filterwarnings("ignore")
filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
filterwarnings("ignore", category=DeprecationWarning) 

data_dir = "./data/bitcoin/processed/"
exp_data_path = "./results/raw/percolation_result.csv"
num_files = 25
max_feat = 21
max_deg = 200

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

def findCorrelations(df):
    from_degree = df['from'].value_counts().to_dict()
    to_degree = df['to'].value_counts().to_dict()

    degree = {key: from_degree.get(key, 0) + to_degree.get(key, 0) for key in set(from_degree) | set(to_degree)}

    max_f, max_deg = df["feature"].max(), max(degree.values())

    pfmk = np.zeros((max_f+1, max_deg+1, max_deg+1))
    for _, row in df.iterrows():
        pfmk[row["feature"]][degree[row["from"]]][degree[row["to"]]] += 1
    pfmk = pfmk / df.shape[0]
    assert abs(np.sum(pfmk) - 1.0) < 1e-5

    return pfmk
    
    """
    from_degree = df['from'].value_counts().to_dict()
    to_degree = df['to'].value_counts().to_dict()

    degree = {key: from_degree.get(key, 0) + to_degree.get(key, 0) for key in set(from_degree) | set(to_degree)}

    df['degreesum'] = df.apply(lambda row: degree[row["from"]]+degree[row["to"]], axis=1)

    degreesum_freq = df['degreesum'].value_counts().to_dict()
    p_sum = lambda s: degreesum_freq[s]/df.shape[0] if s in degreesum_freq else 0

    f_degreesum_freq = {}
    for s in df['degreesum'].unique():
        tmp = df.loc[df['degreesum'] == s]
        f_degreesum_freq[s] = tmp['feature'].value_counts().to_dict()

    p_f_sum = lambda f, s: f_degreesum_freq[s][f]/sum(f_degreesum_freq[s].values()) if s in f_degreesum_freq and f in f_degreesum_freq[s] else 0
    
    return lambda f, s: p_f_sum(f, s), df['degreesum'].max()
    """

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
                sol.x=0.0
                print('raise AssertionError("Solution not found for F0={}".format(F0))')
            else:
                sol.x = sol.x[0]
            if sol.x > 1.0:
                sol.x = 1.0
            self.sol_data.append(1-self.g0(sol.x, degdist, upper_limit))
            if pgr_bar:
                pgr_bar()
    
    def getPlot(self, plt_ax, description, shift=0, scale=1):
        plt_ax.plot((self.bins+shift)/scale, self.exp_data, label='Experimental results')#, linestyle='none', marker='o', fillstyle='none')
        plt_ax.plot((self.bins+shift)/scale, self.sol_data, label="Analytical solution")#, marker='o', fillstyle='none')
        #print(shift, scale, len(self.exp_data))
        if(not self.exp_data):
            return
        #plt_ax.set_xlim(((-0.5+shift)/scale, (len(self.bins)-0.5+shift)/scale))
        plt_ax.set_xlim((-10, 10))
        plt_ax.set_ylim((-0.1, 1.1))
        #plt_ax.set_title(description)#"Uncorrelated features edge percolation \n"+
        #plt_ax.legend(loc='upper left')
        #plt_ax.set_xlabel("Maximum Feature F0")
        #plt_ax.set_ylabel("Size of giant cluster")
        plt_ax.grid(True)
        #return plt

class CorrelatedFeatureEdgePercolation(UncorrelatedFeatureEdgePercolation):

    def __init__(self, net_size, exp_data) -> None:
        super().__init__(net_size, exp_data)
        
    def computeAnalitycalSolutionCorr(self, degdist, pf_mk, pm_k, lower_limit, upper_limit=100, pgr_bar=None):
        def psi(u, k, F0):
            return sum([sum([pf_mk(f, m, k) * pm_k(m, k) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])

        """
        def psi(u, k, F0):
            #a = np.exp(-degdistmean)* sum([(1-u[m-1])*degdistmean**(m-1) / factorial(m-1) * gammaincc(F0, k+m) for m in range(1, upper_limit)])
            #b = sum([sum([excdegdist(m-1) * pfkk(f, (m+k)) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])
            c = sum([(1-u[m-1])* excdegdist(m-1) * gammaincc(F0, (k+m)) for m in range(1, upper_limit)])
            return c
        """

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

        self.sol_data_corr = [0]*len(self.bins)
        with tqdm(total=len(self.bins)) as pbar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val in pool.imap_unordered(func, self.bins):
                    #quit()
                    if(idx==0):
                        val = 0
                    self.sol_data_corr[idx] = val
                    pbar.update()
        #print(self.sol_data_corr)

    def getPlot(self, plt_ax, description, shift=0, scale=1):
        plt_ax.plot((self.bins+shift)/scale, self.exp_data, label='Experimental results')#, linestyle='none', marker='o', fillstyle='none')
        plt_ax.plot((self.bins+shift)/scale, self.sol_data, label="Uncorrelated")#, marker='o', fillstyle='none')
        plt_ax.plot((self.bins+shift)/scale, self.sol_data_corr, label="Correlated")#, marker='o', fillstyle='none')
        #print(shift, scale, len(self.exp_data))
        if(not self.exp_data):
            return
        #plt_ax.set_xlim(((-0.5+shift)/scale, (len(self.bins)-0.5+shift)/scale))
        plt_ax.set_xlim((-10, 10))
        plt_ax.set_ylim((-0.1, 1.1))
        #plt_ax.set_title(description)#"Uncorrelated features edge percolation \n"+
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
    
    with open(exp_data_path) as file:
        exp_data = next(csv.reader(file))

    num_rows, num_cols = 5, 5

    fig_perc, ax_perc = plt.subplots(nrows=num_rows, ncols=num_cols, constrained_layout=True, sharex=True, sharey=True)
    #fig_feat, ax_feat = plt.subplots(nrows=num_rows, ncols=num_cols, constrained_layout=True, sharex=True, sharey=True)

    #with alive_bar(num_files*max_feat, theme='smooth') as bar:
    for i, f in enumerate(sorted(glob.glob(data_dir+"*.txt"))):
        print("Current File Being Processed is: ", i)
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

        pfmk_mat = findCorrelations(df)

        pmk_mat = np.sum(pfmk_mat, axis=0)
        #print(pfmk_mat.shape, pmk_mat.shape)

        pk_mat = np.sum(pmk_mat, axis=0)
        #print(pmk_mat.shape, pk_mat.shape)

        pf_mk = lambda f, m, k: pfmk_mat[f][m][k]/pmk_mat[m][k] if f<pfmk_mat.shape[0] and m<pfmk_mat.shape[1] and k<pfmk_mat.shape[1] and pmk_mat[m][k]>0.0 else 0.0
        pm_k = lambda m, k: pmk_mat[m][k]/pk_mat[k] if m<pfmk_mat.shape[1] and k<pfmk_mat.shape[1] and pk_mat[k]>0.0 else 0.0

        #a = np.array([[pfkk(i, j) for j in range(100)] for i in range(21)])
        #plt.figure()
        #ax_perc[i//num_cols, i%(num_rows)].imshow(a, cmap='hot', interpolation='nearest')
        #plt.show()
        print(pfmk_mat.shape[1])
        
        plotter = CorrelatedFeatureEdgePercolation(nodes, exp_data[i*max_feat:(i+1)*max_feat])#[feat_range[i][0]:feat_range[i][1]])
        plotter.computeAnalitycalSolution(featdist, degdist, excdegdist, max_deg, None)
        plotter.computeAnalitycalSolutionCorr(degdist, pf_mk, pm_k, 0, min(pfmk_mat.shape[1], 50), None)
        #print(f"axperc[{i//num_cols}, {i%(num_rows)}]")
        plotter.getPlot(ax_perc[i//num_cols, i%(num_rows)], "Bitcoin", -10, 1)
        
        #plotter = UncorrelatedFeatureEdgePercolation(nodes, exp_data[i*max_feat:(i+1)*max_feat])#[feat_range[i][0]:feat_range[i][1]])
        #plotter.computeAnalitycalSolution(featdist, degdist, excdegdist, max_deg, None)
        #print(f"axperc[{i//num_cols}, {i%(num_rows)}]")
        #plotter.getPlot(ax_perc[i//num_cols, i%(num_rows)], "Bitcoin", -10, 1)
        #plotdistribution(ax_feat[i//num_cols, i%(1+num_rows)], featdist, 0, max_feat, shift, scale)#-10)

    lines, labels = ax_perc[0, 0].get_legend_handles_labels()
    fig_perc.legend(lines, labels, loc='upper right')

    fig_perc.supxlabel("Maximum Feature F0")
    fig_perc.supylabel("Size of giant cluster")
    fig_perc.suptitle("Uncorrelated features edge percolation - 25 windows \n"+"Bitcoin ALPHA trust weighted signed network")
    """
    fig_feat.supxlabel("Feature value")
    fig_feat.supylabel("Probability")
    fig_feat.suptitle("Feature distribution - 25 windows \n"+"Bitcoin ALPHA trust weighted signed network")
    """
    plt.savefig("./results/figures/bitcoin.pdf")
    plt.show()
    print("plot saved")

if __name__ == "__main__":
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))
    main()