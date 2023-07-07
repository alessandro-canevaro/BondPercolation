from logging import critical
import yaml
import csv
import multiprocess as mp
import numpy as np
from math import asin, sin
from matplotlib import rc
#import matplotlib.pylab as plt
from matplotlib import pyplot as plt
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
    

def computeAnalitycalSolution(degdist, excdegdist, joint_dist, bins, lower_limit=0, upper_limit=10, a=1):

    def pfkk(f, m):
        print(joint_dist(f, m), poisson.pmf(f, m))
        return poisson.pmf(f, m)
    
    def pf(f, m):
        return sum([excdegdist(m-1) * joint_dist(f, m) for m in range(1, upper_limit)])

    def psi(u, k, F0):
        #return sum([(1-u[m-1])* excdegdist(m-1) * gammaincc(F0, 50//(k+m)) for m in range(1, upper_limit)])
        return sum([sum([excdegdist(m-1) * joint_dist(f, m, k) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)])
        return sum([sum([excdegdist(m-1) * joint_dist(f, int(a*(m+k))) * (1-u[m-1]) for m in range(1, upper_limit)]) for f in range(0, F0)]) #for the normal case

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

def plot(exp_data_corr, corr_sol, exp_data_uncorr, uncorr_sol, crit_val_corr, crit_val_uncorr, bins):
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 

    plt.plot(bins,  [0.0, 0.1026957070378389, 0.20085718626253904, 0.3007805365163311, 0.39359209212555957, 0.48027123928461074, 0.5618846904242141, 0.638477226087787, 0.7091668502146844, 0.7733590392353925, 0.8300421618706672, 0.8787014477399627, 0.9184171168942932, 0.9489188599670063, 0.9707215052532053, 0.9845534591496485, 0.9925293405650949, 0.9966531178649882, 0.9985353103748921, 0.999309759877625, 0.9995918526276215],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Neg. Corr. a=50")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins, [0.0, 0.0, 7.997320672359368e-15, 0.0776870512995993, 0.22234118572296369, 0.3735693551247041, 0.5168342022861753, 0.6434603572379312, 0.7485707090994133, 0.8312549313094032, 0.892498631926509, 0.9354233812884217, 0.9635103072267085, 0.9806806841554889, 0.9904818820835773, 0.9955423880624578, 0.9979760888596761, 0.9990586276784044, 0.9994961718872345, 0.9996601716824068, 0.999716067280255],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Neg. Uncorr. a=50")
    plt.axvline(x=2, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins, [0.0, 0.2571440101427628, 0.42852242904972554, 0.5840980636780355, 0.7126657551481604, 0.8159555982169574, 0.8940433598382426, 0.9467330199993084, 0.9770950533091403, 0.9916254726797319, 0.9972845024748884, 0.9991017471903405, 0.9995805212249913, 0.999689300191317, 0.9997101010069707, 0.999713733707318, 0.9997143225310128, 0.9997144103650822, 0.9997144252177469, 0.9997144279647544, 0.9997144286130754],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Neg. Corr. a=25")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    plt.plot(bins, [0.0, 0.027675598696947367, 0.2885827958220923, 0.555815029037744, 0.7522132439666149, 0.8758844347865195, 0.944641688869535, 0.9781107466434048, 0.9922944004042672, 0.9975007966910513, 0.9991439811843683, 0.9995953471346621, 0.9997019023823654, 0.9997243247723396, 0.9997284900750438, 0.9997292305789264, 0.9997293624848168, 0.9997293873168583, 0.9997293932302354, 0.9997293947422077, 0.9997293951688055],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Neg. Uncorr. a=25")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    plt.plot(bins,[0.0, 0.049238324333099825, 0.11573352412116424, 0.18656569901895656, 0.25582152838772304, 0.3231094793556211, 0.3881971511909206, 0.45075869046711536, 0.5107800612701512, 0.5678069750410702, 0.6216495935727392, 0.6718614658877777, 0.7188874512504674, 0.7625396385738955, 0.8031305293606457, 0.8405344884880864, 0.8745026231308054, 0.9049150986977835, 0.9308509233536751, 0.9521516667494878, 0.9687747398935174],    markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Neg. Corr. a=75")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    plt.plot(bins, [0.0, 0.0, 6.9567971383577276e-15, -6.956797138357729e-16, 0.014698869512176712, 0.10737515653281497, 0.2105495297803971, 0.31856570368121806, 0.42511375107700117, 0.5249337037548526, 0.6152698786205715, 0.6943591736896279, 0.7626136464728188, 0.8198931384178352, 0.867100402305553, 0.9049330112543502, 0.9343161177857724, 0.9565109134653537, 0.9723442324899364, 0.9831964298979746, 0.9902771238326971],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Neg. Uncorr. a=75")
    plt.axvline(x=3, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")


    #plt.plot(bins, exp_data_corr,    marker='o', fillstyle='none', linestyle='none', color="#1b9e77", markeredgewidth=2, markersize=6, label='Exp.')
    #plt.plot(bins, exp_data_uncorr,       marker='s', fillstyle='none', linestyle='none', color="#d95f02", markeredgewidth=2, markersize=6, label='Uncorr.')
    #plt.plot(bins, plot_neg_sf,       marker='^', fillstyle='none', linestyle='none', color="#7570b3", markeredgewidth=2, markersize=6, label='Negatively corr.')

    #pos_exp_data_corr = [0.0, 2.9100000000000003e-05, 3.85e-05, 5.33e-05, 8.060000000000001e-05, 0.0001319, 0.00027079999999999997, 0.0009773, 0.2395, 0.727171, 0.8661989999999999, 0.910981, 0.926985, 0.9333410000000001, 0.936174, 0.9375560000000001, 0.938246, 0.938607, 0.938789, 0.9388810000000001, 0.938932]
    #pos_exp_data_uncorr = [0.0, 3.0299999999999998e-05, 4.2999999999999995e-05, 7.14e-05, 0.00012880000000000001, 0.00032399999999999996, 0.0017981, 0.243753, 0.54951, 0.715322, 0.805996, 0.857694, 0.8881739999999999, 0.9068989999999999, 0.918547, 0.925991, 0.930833, 0.933863, 0.935812, 0.937038, 0.937806]
    #plt.plot(bins, pos_exp_data_corr,    marker='o', fillstyle='none', linestyle='none', color="#7570b3", markeredgewidth=2, markersize=6, label='Pos. Corr.')
    #plt.plot(bins, pos_exp_data_uncorr,       marker='s', fillstyle='none', linestyle='none', color="#e7298a", markeredgewidth=2, markersize=6, label='Pos. Uncorr.')
    
    plt.rc('legend', fontsize=12)#, fontsize=12)

    plt.xlim((-0.5, len(bins)-0.5))
    plt.ylim((-0.1, 1.1))
    plt.title("Corr. vs Uncorr. features edge percolation \n SF network a = 3; n=100K; runs=100")
    plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.0, 1))
    plt.xlabel("Maximum Feature F0")
    plt.ylabel("Size of giant cluster")
    plt.xticks(list(range(0, len(bins)+1, 2)))
    return plt

def criticalpoint(bins, excdegdist, jointdist, upper_limit=50, a=1):
    eig_vals = []
    for F0 in range(len(bins)):
        G = np.zeros((upper_limit, upper_limit))
        for i in range(upper_limit):
            for j in range(upper_limit):
                G[i, j] = (i)*excdegdist(i)*sum([jointdist(f, int(a*((i+1)+(j+1)))) for f in range(0, F0)])
        eigenvalues, eigenvectors = eig(G)
        eig_vals.append(max(eigenvalues.real.tolist()))
    
    eig_vals = np.array(eig_vals)
    eig_vals[eig_vals > 1] = 0
    crit_val = np.argmax(eig_vals)
    return crit_val

def generate_distributions(a, lower_limit, upper_limit, feature_limit=21):
    degdist = lambda k: binom.pmf(k, 100000, 0.00003)

    degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
    def excdegdist(k):
        return degdist(k+1)*(k+1)/degdistmean
    
    jointdist = lambda k, m: poisson.pmf(k, m)

    uncorr_joint_dist_list = [sum([sum([excdegdist(m-1)*excdegdist(k-1) * jointdist(f, int(a*(m+k))) for m in range(1, upper_limit)]) for k in range(1, upper_limit)]) for f in range(feature_limit)]
    uncorr_jointdist = lambda k, m: uncorr_joint_dist_list[k]
    
    return degdist, excdegdist, jointdist, uncorr_jointdist

def epsilon_metric():
    a_list = [3]# np.arange(0, 155, 5)
    #a_list = np.arange(0, 3, 1)

    def func(a):
        bins = np.arange(0, 31)
        lower_limit = 0
        upper_limit = 20
        degdist, excdegdist, jointdist, uncorr_jointdist = generate_distributions(a, lower_limit, upper_limit, feature_limit=31)
        
        corr_data = computeAnalitycalSolution(degdist, excdegdist, jointdist, bins, lower_limit, upper_limit, a)
        uncorr_data = computeAnalitycalSolution(degdist, excdegdist, uncorr_jointdist, bins, lower_limit, upper_limit, a)
        print(corr_data, uncorr_data)

        crit_point_corr = criticalpoint(bins, excdegdist, jointdist, upper_limit, a)
        crit_point_uncorr = criticalpoint(bins, excdegdist, uncorr_jointdist, upper_limit, a)

        crit_point_min = min(crit_point_corr, crit_point_uncorr)
        s_diff_sum = np.sum(np.array(corr_data[crit_point_min:])-np.array(uncorr_data[crit_point_min:]))
        print(a, crit_point_corr, crit_point_uncorr, s_diff_sum)
        return a, s_diff_sum

    """
    sol_data = []
    with tqdm(total=len(a_list)) as pbar:
        #with mp.Pool(mp.cpu_count()) as pool:
        for a in a_list:
            sol_data.append(func(a))
            pbar.update()
    sol_data = sorted(sol_data, key=lambda x: x[0])
    print(sol_data)
    """

    pos_data = [(0, 0.0), (0.5, -0.03415858822768275), (1, -0.3422681836206113), (1.5, -0.4878888857224688), (2, -0.8123369407319166), (2.5, -1.0026445186108425), (3, -1.1497365229028107)]
    neg_data = [(0, 0.0), (25, 0.027785942337264558), (50, 0.16335445243700408), (75, 0.3232209637181103), (100, 0.6038037386873321), (125, 1.0221037603323224), (150, 1.5243232014921855)]

    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    lns1 = ax1.plot(*zip(*neg_data), marker="o", markeredgewidth=2, markersize=6, mfc='none', color="#1b9e77", label="neg. corr")
    ax1.set_xlabel(r"neg. corr. coeff. a-")
    lns2 = ax2.plot(*zip(*pos_data), marker="s", markeredgewidth=2, markersize=6, mfc='none', color="#d95f02", label="pos. corr")
    #lns3 = ax2.plot(*zip(*sol_data), marker="x", markeredgewidth=2, markersize=6, mfc='none', label="sol data")
    ax2.set_xlabel(r"pos. corr. coeff. a+")
    ax1.set_ylabel(r"Epsilon")
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")
    plt.savefig("./results/figures/epsilon.pdf", bbox_inches='tight')


def deltaF0():

    a_list = np.arange(0, 155, 5)
    #a_list = np.arange(0, 3.1, 0.1)

    def func(a):
        bins = np.arange(0, 41)
        lower_limit = 0
        upper_limit = 20
        degdist, excdegdist, jointdist, uncorr_jointdist = generate_distributions(a, lower_limit, upper_limit, feature_limit=41)
        
        crit_point_corr = criticalpoint(bins, excdegdist, jointdist, upper_limit, a)
        crit_point_uncorr = criticalpoint(bins, excdegdist, uncorr_jointdist, upper_limit, a)
        return a, crit_point_corr - crit_point_uncorr

    
    sol_data = []
    with tqdm(total=len(a_list)) as pbar:
        with mp.Pool(mp.cpu_count()) as pool:
            for val in pool.imap_unordered(func, a_list):
                sol_data.append(val)
                pbar.update()
    sol_data = sorted(sol_data, key=lambda x: x[0])
    print(sol_data)
    

    
    pos_data = [(0.0, 0), (0.1, 0), (0.2, 1), (0.30000000000000004, 0), (0.4, 0), (0.5, 0), (0.6000000000000001, 1), (0.7000000000000001, 0), (0.8, 1), (0.9, 1), (1.0, 1), (1.1, 1), (1.2000000000000002, 1), (1.3, 1), (1.4000000000000001, 2), (1.5, 2), (1.6, 1), (1.7000000000000002, 2), (1.8, 2), (1.9000000000000001, 2), (2.0, 2), (2.1, 2), (2.2, 2), (2.3000000000000003, 3), (2.4000000000000004, 3), (2.5, 2), (2.6, 3), (2.7, 3), (2.8000000000000003, 3), (2.9000000000000004, 3), (3.0, 3)]
    neg_data = [(0, 0), (5, 0), (10, 0), (15, -1), (20, 0), (25, -1), (30, 0), (35, -1), (40, -2), (45, -1), (50, -2), (55, -1), (60, -2), (65, -1), (70, -2), (75, -2), (80, -2), (85, -3), (90, -2), (95, -2), (100, -2), (105, -3), (110, -3), (115, -3), (120, -4), (125, -4), (130, -4), (135, -4), (140, -4), (145, -4), (150, -4)]

    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    
    lns1 = ax1.plot(*zip(*neg_data), marker="o", markeredgewidth=2, markersize=6, mfc='none', color="#1b9e77", label="neg. corr")
    ax1.set_xlabel(r"neg. corr. coeff. a-")
    lns2 = ax2.plot(*zip(*pos_data), marker="s", markeredgewidth=2, markersize=6, mfc='none', color="#d95f02", label="pos. corr")
    #lns3 = ax2.plot(*zip(*sol_data), marker="x", markeredgewidth=2, markersize=6, mfc='none', label="sol data")
    ax2.set_xlabel(r"pos. corr. coeff. a+")
    ax1.set_ylabel(r"Delta F0 Crit.")
    ax1.yaxis.grid()
    ax1.xaxis.grid()
    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")
    plt.savefig("./results/figures/delta_F0.pdf", bbox_inches='tight')

def other_dist():
    def generate_dist(a, lower_limit, upper_limit, feature_limit=21):
        degdist = lambda k: binom.pmf(k, 100000, 0.00003)
        #C = sum([ks**(-3) for ks in range(lower_limit, upper_limit) if ks>=2])
        #degdist = lambda k: (k**(-3)/C) if k>=2 else 0

        degdistmean = sum([k*degdist(k) for k in range(lower_limit, upper_limit)])
        def excdegdist(k):
            return degdist(k+1)*(k+1)/degdistmean
        
        #jointdist = lambda f, k, m: poisson.pmf(f, a*(m+k)) if m+k>0 else np.nan
        #_jointdist = lambda f, k, m: a * (1+k+m)**a / (f+k+m)**(1+a) if f+m+k > 0 else 0
        _jointdist = lambda f, k, m: a * (k+m)*(1+k+m)**a / (f*(k+m)+1)**(1+a)
        
        data = np.zeros((feature_limit, (upper_limit*2)+1))
        for f in range(feature_limit):
            for d in range((upper_limit*2)+1):
                data[f, d] = _jointdist(f, d, 0)
        norm = np.sum(data, axis=0)
        jointdist = lambda f, k, m: _jointdist(f, k, m)/norm[m+k]

        uncorr_joint_dist_list = [sum([sum([excdegdist(m-1)*excdegdist(k-1) * jointdist(f, k, m) for m in range(1, upper_limit)]) for k in range(1, upper_limit)]) for f in range(feature_limit)]
        uncorr_jointdist = lambda f, k, m: uncorr_joint_dist_list[f]
        
        return degdist, excdegdist, jointdist, uncorr_jointdist
    
    def plot_jointdist(jointdist):
        feat = 21
        deg = 20
        data = np.zeros((feat, deg))
        for f in range(feat):
            for d in range(deg):
                data[f, d] = jointdist(f, d, 0)
        norm = np.sum(data, axis=0)
        #print(norm)
        #for d in range(deg):
        #    data[:, d] = data[:, d] / norm[d]
        matplotlib.rcParams.update({'font.size': 18})
        matplotlib.rc('xtick', labelsize=16) 
        matplotlib.rc('ytick', labelsize=16) 
        plt.imshow(data, cmap="inferno")
        plt.colorbar()
        plt.title("Conditional feature dist.")
        plt.xlabel("Degree sum k+m")
        plt.ylabel("Feature f")
        plt.savefig("./results/figures/PMA_POS.pdf", bbox_inches='tight')
        plt.clf()
        plt.close()

    
    def criticalpoint(bins, excdegdist, jointdist, upper_limit=50, a=1):
        eig_vals = []
        for F0 in range(len(bins)):
            G = np.zeros((upper_limit, upper_limit))
            for i in range(upper_limit):
                for j in range(upper_limit):
                    G[i, j] = (i)*excdegdist(i)*sum([jointdist(f, (i+1), (j+1)) for f in range(0, F0)])
            eigenvalues, eigenvectors = eig(G)
            eig_vals.append(max(eigenvalues.real.tolist()))
        
        eig_vals = np.array(eig_vals)
        eig_vals[eig_vals > 1] = 0
        crit_val = np.argmax(eig_vals)
        return crit_val


    a_list = np.arange(0, 155, 5)
    #a_list = np.arange(0, 3.1, 0.1)
    a = 0.005

    bins = np.arange(0, 21)
    lower_limit = 0
    upper_limit = 20
    #upper_limit = 100#int(100000**0.5)+1

    degdist, excdegdist, jointdist, uncorr_jointdist = generate_dist(a, lower_limit, upper_limit, feature_limit=21)
    #plot_jointdist(jointdist)
    
    """
    crit_point_corr = criticalpoint(bins, excdegdist, jointdist, upper_limit, a)
    crit_point_uncorr = criticalpoint(bins, excdegdist, uncorr_jointdist, upper_limit, a)
    print(crit_point_corr, crit_point_uncorr)
    
    corr_sol = computeAnalitycalSolution(degdist, excdegdist, jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    print("corr sol", corr_sol, crit_point_corr)
    uncorr_sol = computeAnalitycalSolution(degdist, excdegdist, uncorr_jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    print("uncorr sol", uncorr_sol, crit_point_uncorr)
    """

    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    #plt.plot(bins,  corr_sol,    markersize=8, linestyle='dashed', color="k", marker='o', label="test. Corr. ")
    #plt.plot(bins,  uncorr_sol,    markersize=8, linestyle='dashed', color="k", marker='s', label="test. UnCorr. ")
    
    plt.plot(bins,  [0.0, 0.936885836159678, 0.939354338435475, 0.9399106967188803, 0.9401314539896946, 0.9402441226126497, 0.9403104706043864, 0.9403533484512049, 0.9403829213462798, 0.9404043239729523, 0.9404203983165638, 0.9404332942333421, 0.9404431447154211, 0.940451105384514, 0.9404576473747955, 0.9404631004307346, 0.9404677036230079, 0.940471630342822, 0.9404750123120468, 0.9404779497374227, 0.9404805203173586],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Corr. a=1.5")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins, [0.0, 0.9387359829463515, 0.9399602968633217, 0.9402217599663478, 0.9403235299671916, 0.940374954974941, 0.940405079980625, 0.9404244599775451, 0.9404377883085916, 0.9404474137044627, 0.9404546309891667, 0.9404602062025003, 0.9404646183973948, 0.9404681808690998, 0.9404711062151453, 0.9404740034616456, 0.9404760587241452, 0.9404778112601375, 0.9404793200024042, 0.9404806299134806, 0.9404817758321937],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Uncorr. a=1.5")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins, [0.0, 0.7988671011511704, 0.8502016218152636, 0.87258921537731, 0.8861491400331288, 0.8955948331352072, 0.9027104024218099, 0.9083466863515226, 0.9129702260463111, 0.9168621426256958, 0.9202038009486527, 0.923118433948199, 0.9256932579979318, 0.9279920639369097, 0.9300627923894191, 0.9319423019703342, 0.933659485804641, 0.9352373745761483, 0.9366945951193147, 0.938046406408833, 0.9393054509047535],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Corr. a=0.005")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    plt.plot(bins, [0.0, 0.8158703876459944, 0.8647978436642012, 0.8846674091977768, 0.8963057252202494, 0.9042489677246747, 0.9101495588303821, 0.9147757185919833, 0.918540880155902, 0.9216905654151604, 0.9243813215607465, 0.9267184858479786, 0.9287759956579749, 0.9306075461253743, 0.932253241148056, 0.93374374609053, 0.9351029827094883, 0.9363499361772223, 0.9374999012178373, 0.9385653626292367, 0.9395566308751289],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Uncorr. a=0.005")
    plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins,[0.0, 0.46315606220610334, 0.8512875451702445, 0.9078282467798547, 0.9246638841613168, 0.9316257236397262, 0.9350756769521023, 0.9369877514631645, 0.9381308612218616, 0.9388528684260034, 0.939328495152825, 0.9396525015481045, 0.9398793900828932, 0.9400420018155137, 0.9401608904895888, 0.940249333387481, 0.9403161413730885, 0.9403672989994638, 0.940406954789432, 0.9404380368750833, 0.9404626461459237],    markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Corr. a=5")
    #plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    #plt.plot(bins, [0.0, 0.550606045655423, 0.8241182719588069, 0.8857831893578532, 0.9095919592074404, 0.9212392174334446, 0.9277446044447327, 0.9316981169506615, 0.9342457460109734, 0.9359600588221776, 0.9371529851503713, 0.93800579350738, 0.9386292173305624, 0.9390936502270971, 0.9394453237579654, 0.939715439227877, 0.9399255481307134, 0.9400908380786986, 0.9402222016606993, 0.940327574084523, 0.9404128322850048],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Uncorr. a=5")
    #plt.axvline(x=0, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")
 
    plt.rc('legend', fontsize=12)#, fontsize=12)

    plt.xlim((-0.5, len(bins)-0.5))
    plt.ylim((-0.1, 1.1))
    plt.title("Negative Feature-Degree Correlation \nER network")
    plt.legend(loc='lower right')#, frameon=False, bbox_to_anchor=(1.0, 1))
    plt.xlabel("Maximum Feature F0")
    plt.ylabel("Size of giant cluster")
    plt.xticks(list(range(0, len(bins)+1, 2)))
    plt.savefig("./results/figures/PLC_ER_NEG.pdf", bbox_inches='tight')

def main():
    lower_limit = 0
    upper_limit = 10
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

    uncorr_joint_dist_list = [sum([sum([excdegdist(m-1)*excdegdist(k-1) * jointdist(f, int(75//(m+k))) for m in range(1, upper_limit)]) for k in range(1, upper_limit)]) for f in range(21)]
    uncorr_jointdist = lambda k, m: uncorr_joint_dist_list[k]

    crit_point_corr = criticalpoint(bins, excdegdist, jointdist, upper_limit)
    crit_point_uncorr = criticalpoint(bins, excdegdist, uncorr_jointdist, upper_limit)

    corr_sol = computeAnalitycalSolution(degdist, excdegdist, jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    uncorr_sol = computeAnalitycalSolution(degdist, excdegdist, uncorr_jointdist, bins, lower_limit=0, upper_limit=upper_limit)
    print("corr sol", corr_sol, crit_point_corr)
    #print("corr exp", exp_data_corr)
    print("uncorr sol", uncorr_sol, crit_point_uncorr)
    #print("uncorr exp", exp_data_uncorr)

    plt = plot(exp_data_corr, corr_sol, exp_data_uncorr, uncorr_sol, crit_point_corr, crit_point_uncorr, bins)
    #plt.tight_layout()
    plt.savefig("./results/figures/SF_NEG.pdf", bbox_inches='tight')

if __name__ == "__main__":
    #main()
    other_dist()

    
    

    
