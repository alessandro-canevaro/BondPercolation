from logging import critical
import yaml
import csv
import multiprocess as mp
import numpy as np
from math import asin, sin
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.pylab as plt
import matplotlib
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})#, 'size': 14
rc('text', usetex=True)

from scipy import optimize
from scipy.linalg import eig
from scipy.stats import binom, geom, poisson
from scipy.misc import derivative
from scipy.special import factorial, gammaincc#, gamma, gammaincc
from mpmath import gamma, gammainc
from alive_progress import alive_bar
from warnings import filterwarnings
from tqdm import tqdm

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

class UncorrelatedFeatureEdgePercolation(NodeUniformRemoval):
    
    def __init__(self, net_size, data_path) -> None:
        super().__init__(net_size, data_path)
        self.bins = np.arange(0, len(self.exp_data))

    def computeAnalitycalSolution(self, degdist, excdegdist, lower_limit, upper_limit):
        self.sol_data = []
        with tqdm(total=len(self.bins)) as pbar:
            for F0 in self.bins:
                psi = sum([poisson.pmf(f, 8) for f in range(0, F0)])
                def f(u):
                    return u - 1 + psi - psi * self.g1(u, excdegdist, lower_limit, upper_limit)

                sol = optimize.root(f, 0)
                if not sol.success:
                    raise AssertionError("Solution not found for F0={}".format(F0))
                self.sol_data.append(1-self.g0(sol.x[0], degdist, lower_limit, upper_limit))
                pbar.update()
    
    def getPlot(self, description):
        
        matplotlib.rcParams.update({'font.size': 18})
        matplotlib.rc('xtick', labelsize=16) 
        matplotlib.rc('ytick', labelsize=16) 
        print(self.sol_data)
        data_uncorr = [-1.2678746941219288e-13, -1.2678746941219288e-13, -1.2634338020234281e-13, -1.2678746941219288e-13, -1.2678746941219288e-13, -7.793765632868599e-14, -1.2678746941219288e-13, -1.312283615106935e-13, 0.4770191571193989, 0.7238158542797648, 0.8332841816477693, 0.8855377849838402, 0.9119288406535522, 0.9257293677530405, 0.9330246746269923, 0.9368375003900211, 0.9387733561659866, 0.939716895649021, 0.9401554119780537, 0.9403491898332481, 0.9404305756312835]
        data_pos = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22956353297666685, 0.7255128745229571, 0.8668318518028086, 0.9121645901192609, 0.9283579168016801, 0.934804667974974, 0.9376574623430802, 0.9390325776813758, 0.9397313798027581, 0.9400947417141623, 0.9402843129052768, 0.9403824967824943, 0.9404327209510531]
        data_neg = [0, 0.0, 0.0, 0.0, 0.15009389964058042, 0.3404145111437191, 0.5190121477116831, 0.6395240213116969, 0.722307658377451, 0.7803228477257323, 0.8217683401040111, 0.8519155046702866, 0.8741792865069385, 0.890821615615358, 0.903385022548715, 0.9129413519534159, 0.9202399474789416, 0.9258081077779224, 0.9300259070262958, 0.9331795277018256, 0.9354958847255173]
        
        data_uncorr_sf = [6.661338147750939e-16, -1.6562655247298608e-08, -1.5205720504951614e-07, -7.532048174052619e-07, -3.025525771915838e-06, -1.8117590261601535e-05, 0.05053291809678895, 0.23887629709498215, 0.4928585746797187, 0.7149897370612075, 0.8625759552612297, 0.9424556094425138, 0.9789012907948944, 0.9931854414616883, 0.9980514259678364, 0.9995044392804303, 0.9998874000336544, 0.9999770392402455, 0.9999957795926181, 0.9999992977411841, 0.9999998937845268]
        data_pos_sf = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.094846423568335e-16, 6.513511428367715e-16, 0.3623820674947328, 0.7027089866802432, 0.834397586530612, 0.8943224231465063, 0.9256666743625444, 0.9440822620641341, 0.9559551467971449, 0.9641733641680936, 0.9701700559309545, 0.9747196306744533, 0.97827383180153, 0.9811140438572712]
        data_neg_sf = [0, 0.10961051652787747, 0.20605711773617089, 0.30406802146449885, 0.3953359602899987, 0.4807840158104402, 0.5613080899431969, 0.636786897801514, 0.7066190951305261, 0.7700899228359587, 0.826448359485246, 0.8749006911034141, 0.9147013473418749, 0.945454130343778, 0.9674684965391132, 0.9818962118536543, 0.9905004948361259, 0.995178710128654, 0.9975245550792605, 0.9986330290695927, 0.9991411719091038]
        plt.plot(self.bins, data_uncorr_sf,    markersize=8, linestyle='dashed', color="gray", marker='.', label="Analytical solution")
        plt.plot(self.bins, data_pos_sf,       markersize=8, linestyle='dashed', color="gray", marker='.')
        plt.plot(self.bins, data_neg_sf,       markersize=8, linestyle='dashed', color="gray", marker='.')

        print(self.exp_data)
        
        plot_uncorr = [0.0, 2e-05, 3.08e-05, 4.2699999999999994e-05, 6.87e-05, 0.0001341, 0.00038439999999999997, 0.011251800000000001, 0.46986900000000004, 0.7148639999999999, 0.8278160000000001, 0.882084, 0.9095369999999999, 0.923841, 0.93136, 0.9352889999999999, 0.937329, 0.938322, 0.938786, 0.9389860000000001, 0.939076]
        plot_pos = [0.0, 2.88e-05, 3.88e-05, 5.4199999999999996e-05, 8.07e-05, 0.00013220000000000001, 0.0002745, 0.0009565999999999999, 0.23437, 0.727139, 0.866385, 0.9110910000000001, 0.9269689999999999, 0.933414, 0.936233, 0.9376289999999999, 0.9383, 0.9386660000000001, 0.938847, 0.9389419999999999, 0.938987]
        plot_neg = [0.0, 5.12e-05, 0.00016570000000000002, 0.0013689, 0.11128, 0.331378, 0.505996, 0.628614, 0.7141139999999999, 0.774086, 0.817022, 0.8483010000000001, 0.871359, 0.888491, 0.901517, 0.9113110000000001, 0.918737, 0.924386, 0.928695, 0.931853, 0.9341689999999999]
        
        plot_uncorr_sf = [0.0, 2.18e-05, 3.44e-05, 7.280000000000001e-05, 0.00023899999999999998, 0.0047079, 0.0503518, 0.21197, 0.46436900000000003, 0.696778, 0.85364, 0.938728, 0.977567, 0.99275, 0.997899, 0.99947, 0.999872, 0.999972, 0.9999910000000001, 0.9999939999999999, 0.999995]
        plot_pos_sf = [0.0, 3.0299999999999998e-05, 4.2300000000000005e-05, 6.36e-05, 9.470000000000001e-05, 0.000157, 0.0002953, 0.0007484999999999999, 0.0046198, 0.43474300000000005, 0.728106, 0.843579, 0.897711, 0.9271710000000001, 0.9448510000000001, 0.9563980000000001, 0.9644410000000001, 0.97047, 0.975043, 0.97856, 0.981394]
        plot_neg_sf = [0.0, 0.120952, 0.20810700000000001, 0.299357, 0.38616900000000004, 0.469101, 0.548598, 0.624159, 0.695091, 0.760289, 0.8187089999999999, 0.8693989999999999, 0.9113669999999999, 0.944096, 0.967707, 0.982955, 0.991836, 0.996474, 0.99863, 0.999511, 0.9998360000000001]
        plt.plot(self.bins, plot_uncorr_sf,    marker='o', fillstyle='none', linestyle='none', color="#1b9e77", markeredgewidth=2, markersize=6, label='Uncorr.')
        plt.plot(self.bins, plot_pos_sf,       marker='s', fillstyle='none', linestyle='none', color="#d95f02", markeredgewidth=2, markersize=6, label='Positevely corr.')
        plt.plot(self.bins, plot_neg_sf,       marker='^', fillstyle='none', linestyle='none', color="#7570b3", markeredgewidth=2, markersize=6, label='Negatively corr.')
        
        plt.rc('legend', fontsize=12)#, fontsize=12)

        plt.xlim((-0.5, len(self.bins)-0.5))
        plt.ylim((-0.1, 1.1))
        #plt.title("Uncorrelated features edge percolation \n"+description)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("Maximum Feature F0")
        plt.ylabel("Size of giant cluster")
        plt.xticks(list(range(0, len(self.bins)+1, 2)))
        #plt.grid(True)
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

        upper_limit = 100

        def pfkk(f, m):
            return poisson.pmf(f, m)

        def psi(u, k, F0):
            #a = np.exp(-degdistmean)* sum([(1-u[m-1])*degdistmean**(m-1) / factorial(m-1) * gammaincc(F0, k+m) for m in range(1, upper_limit)])
            #b = sum([sum([excdegdist(m-1) * pfkk(f, (m+k)) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])
            c = sum([(1-u[m-1])* excdegdist(m-1) * gammaincc(F0, 50//(k+m)) for m in range(1, upper_limit)])
            return c

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
        with tqdm(total=len(self.bins)) as pbar:
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, val in pool.imap_unordered(func, self.bins):
                    if(idx==0):
                        val = 0
                    self.sol_data[idx] = val
                    pbar.update()


    def getPlot(self, description):
        plt = super().getPlot(description)
        plt.title("Correlated features edge percolation \n"+description)
        plt.xlabel("Maximum Feaure F0 - P(F | k, k') = Poisson(a//(k+k'))")
        return plt
    

def computeAnalitycalSolution(degdist, excdegdist, joint_dist, bins, lower_limit=0, upper_limit=10):

    def pfkk(f, m):
        print(joint_dist(f, m), poisson.pmf(f, m))
        return poisson.pmf(f, m)
    
    def pf(f, m):
        return sum([excdegdist(m-1) * joint_dist(f, m) for m in range(1, upper_limit)])

    def psi(u, k, F0):
        #return sum([(1-u[m-1])* excdegdist(m-1) * gammaincc(F0, 50//(k+m)) for m in range(1, upper_limit)])
        return sum([sum([excdegdist(m-1) * joint_dist(f, 50//(m+k)) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])

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

    sol_data = [0]*len(bins)
    with tqdm(total=len(bins)) as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for idx, val in pool.imap_unordered(func, bins):
                sol_data[idx] = val
                pbar.update()
    return sol_data

def plot(exp_data_corr, corr_sol, exp_data_uncorr, uncorr_sol, bins):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 

    plt.plot(bins, corr_sol,    markersize=8, linestyle='dashed', color="gray", marker='.', label="Analytical solution")
    plt.plot(bins, uncorr_sol,       markersize=8, linestyle='dashed', color="gray", marker='.')

    pos_corr_sol = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.409814487374206e-16, 0.2321158969459163, 0.7261132196671727, 0.8669347079606387, 0.9121400231551087, 0.9283158705985243, 0.9347521444010407, 0.9375880143371299, 0.9389590027544457, 0.9396570929453226, 0.9400188035169329, 0.9402076320439814, 0.9403058790597177, 0.9403566498138933]
    pos_uncorr_sol = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2290688437191313, 0.5590733894578962, 0.7240345243062353, 0.8121791082222132, 0.8619532873322828, 0.8914101556342454, 0.9094241763489905, 0.9207193797387009, 0.927913797899869, 0.9325293391148012, 0.9354855609014825, 0.9373671155659784, 0.9385586601665219, 0.9393001932415312]
    plt.plot(bins, pos_corr_sol,    markersize=8, linestyle='dashed', color="gray", marker='.')
    plt.plot(bins, pos_uncorr_sol,       markersize=8, linestyle='dashed', color="gray", marker='.')
    #plt.plot(bins, data_neg_sf,       markersize=8, linestyle='dashed', color="gray", marker='.')

    plt.plot(bins, exp_data_corr,    marker='o', fillstyle='none', linestyle='none', color="#1b9e77", markeredgewidth=2, markersize=6, label='Neg. Corr.')
    plt.plot(bins, exp_data_uncorr,       marker='s', fillstyle='none', linestyle='none', color="#d95f02", markeredgewidth=2, markersize=6, label='Neg. Uncorr.')
    #plt.plot(bins, plot_neg_sf,       marker='^', fillstyle='none', linestyle='none', color="#7570b3", markeredgewidth=2, markersize=6, label='Negatively corr.')

    pos_exp_data_corr = [0.0, 2.9100000000000003e-05, 3.85e-05, 5.33e-05, 8.060000000000001e-05, 0.0001319, 0.00027079999999999997, 0.0009773, 0.2395, 0.727171, 0.8661989999999999, 0.910981, 0.926985, 0.9333410000000001, 0.936174, 0.9375560000000001, 0.938246, 0.938607, 0.938789, 0.9388810000000001, 0.938932]
    pos_exp_data_uncorr = [0.0, 3.0299999999999998e-05, 4.2999999999999995e-05, 7.14e-05, 0.00012880000000000001, 0.00032399999999999996, 0.0017981, 0.243753, 0.54951, 0.715322, 0.805996, 0.857694, 0.8881739999999999, 0.9068989999999999, 0.918547, 0.925991, 0.930833, 0.933863, 0.935812, 0.937038, 0.937806]
    plt.plot(bins, pos_exp_data_corr,    marker='o', fillstyle='none', linestyle='none', color="#7570b3", markeredgewidth=2, markersize=6, label='Pos. Corr.')
    plt.plot(bins, pos_exp_data_uncorr,       marker='s', fillstyle='none', linestyle='none', color="#e7298a", markeredgewidth=2, markersize=6, label='Pos. Uncorr.')
    
    plt.rc('legend', fontsize=12)#, fontsize=12)

    plt.xlim((-0.5, len(bins)-0.5))
    plt.ylim((-0.1, 1.1))
    plt.title("Corr. vs Uncorr. features edge percolation \n ER network k = 3; n=100K; runs=100")
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("Maximum Feature F0")
    plt.ylabel("Size of giant cluster")
    plt.xticks(list(range(0, len(bins)+1, 2)))
    return plt

def criticalpoint(bins, excdegdist, jointdist, upper_limit=10):
    upper_limit = 10
    G = np.zeros((upper_limit, upper_limit))
    for i in range(upper_limit):
        for j in range(upper_limit):
            G[i, j] = (i)*excdegdist(i)*sum([jointdist(f, 50//((i+1)+(j+1))) for f in range(0, len(bins))])
    eigenvalues, eigenvectors = eig(G)
    print(eigenvalues.shape, eigenvectors.shape)

def main():
    lower_limit = 0
    upper_limit = 50
    data_dir = "./results/raw/"

    #load configuration file
    with open("./experiments/test.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        net_size = int(params['network_size'])
        net_type = params['network_type']
        perc_type = params['percolation_type']
        param1 = float(params['param1'])

    #load experimental data (corr)
    with open(data_dir+"perc_result_corr.csv") as file:
        exp_data_corr = next(csv.reader(file))
    exp_data_corr = [float(i)/net_size for i in exp_data_corr]
    bins = np.arange(0, len(exp_data_corr))

    #load experimental data (uncorr)
    with open(data_dir+"perc_result_uncorr.csv") as file:
        exp_data_uncorr = next(csv.reader(file))
    exp_data_uncorr = [float(i)/net_size for i in exp_data_uncorr]

    #load deg dist
    with open(data_dir+"perc_result_degdist.csv") as file:
        deg_dist_list = next(csv.reader(file))
    degdist = lambda k: float(deg_dist_list[k])

    degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
    print("deg. dist. mean:", degdistmean)

    def excdegdist(k):
        return degdist(k+1)*(k+1)/degdistmean
    
    #load joint dist:
    with open(data_dir+"perc_result_jointdist.csv") as file:
        joint_dist_list = next(csv.reader(file))
    jointdist = lambda k, m: float(joint_dist_list[21*m+k]) if m < 200 and k < 21 else print("mk", m, k)


    #criticalpoint(bins, excdegdist, jointdist)

    uncorr_joint_dist_list = [sum([sum([excdegdist(m-1)*excdegdist(k-1) * jointdist(f, 50//(m+k)) for m in range(1, upper_limit)]) for k in range(1, upper_limit)]) for f in range(21)]
    uncorr_jointdist = lambda k, m: uncorr_joint_dist_list[k]

    corr_sol = computeAnalitycalSolution(degdist, excdegdist, jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    uncorr_sol = computeAnalitycalSolution(degdist, excdegdist, uncorr_jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    print("corr sol", corr_sol)
    print("corr exp", exp_data_corr)
    print("uncorr sol", uncorr_sol)
    print("uncorr exp", exp_data_uncorr)

    plt = plot(exp_data_corr, corr_sol, exp_data_uncorr, uncorr_sol, bins)
    plt.savefig("./results/figures/ER_corr_vs_uncorr.pdf")

if __name__ == "__main__":
    main()

    
    

    
