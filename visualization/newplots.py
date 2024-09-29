import itertools
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
from scipy.special import factorial, gammaincc, zeta#, gamma, gammaincc
from mpmath import gamma, gammainc
from alive_progress import alive_bar
from warnings import filterwarnings
from tqdm import tqdm

filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


def computeAnalitycalSolution(degdist, excdegdist, conddist, bins, upper_limit=10):
        
    def psi(u, k, F0):
        return sum([sum([excdegdist(l-1) * conddist(f, k, l) * (1-u[l-1]) for l in range(1, upper_limit)]) for f in range(1, F0)])

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

def generate_dist(beta, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim):

    #degdist = lambda k: binom.pmf(k, 100000, 0.00003)
    C = sum([ks**(-4) for ks in range(deg_low_lim, deg_high_lim) if ks>=2])
    degdist = lambda k: (k**(-4)/C) if k>=2 else 0

    degdistmean = sum([k*degdist(k) for k in range(deg_low_lim, deg_high_lim)])

    excdegdist = lambda k: degdist(k+1)*(k+1)/degdistmean
            
    joint_deg_dist = lambda k, l: excdegdist(k-1)*excdegdist(l-1) if deg_low_lim<=k<deg_high_lim and deg_low_lim<=l<deg_high_lim else 0.0
    assert abs(sum([sum([joint_deg_dist(k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) - 1) < 0.01, "joint_deg_dist is not normalized"
    
    #_conddist = lambda f, k, l: (f+k+l)**(-3-beta)*zeta(3 + beta, 1 + k + l)**(-1)
    _conddist = lambda f, k, l: (k+l)**(2+beta) * zeta(2 + beta, (1 + k + l)/(k + l))**(-1) * (1+f*(k+l))**(-2-beta) if k+l>0 else 0.0
    norm = sum([sum([sum([joint_deg_dist(k, l) * _conddist(f, k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) for f in range(feat_low_lim, feat_high_lim)])
    conddist = lambda f, k, l: _conddist(f, k, l)/norm if deg_low_lim<=k<deg_high_lim and deg_low_lim<=l<deg_high_lim else 0.0

    assert abs(sum([sum([sum([joint_deg_dist(k, l) * conddist(f, k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) for f in range(feat_low_lim, feat_high_lim)]) - 1) < 0.01, "cond_dist is not normalized"

    uncorr_cond_dist_list = [sum([sum([joint_deg_dist(k, l) * conddist(f, k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) for f in range(feat_low_lim, feat_high_lim)]
    uncorr_conddist = lambda f, k, l: uncorr_cond_dist_list[f-feat_low_lim]
    assert abs(sum([sum([sum([joint_deg_dist(k, l) * uncorr_conddist(f, k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) for f in range(feat_low_lim, feat_high_lim)]) - 1) < 0.01, "uncorr_conddist is not normalized"

    return degdist, excdegdist, joint_deg_dist, conddist, uncorr_conddist


def plot_conddist(conddist, joint_deg_dist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim):
    all_rolls = list(itertools.product(range(deg_high_lim), repeat=2))
    combinations = [(t_sum, [(die1, die2) for die1, die2 in all_rolls if die1 + die2 == t_sum]) for t_sum in range(deg_low_lim*2, deg_high_lim*2)]

    d_dist = lambda d: sum([joint_deg_dist(k, l) for k, l in combinations[d-deg_low_lim*2][1]])
    assert abs(sum([d_dist(d) for d in range(deg_low_lim*2, deg_high_lim*2)]) - 1) < 0.01, "d_dist is not normalized"

    join_d_dist = lambda f, d: sum([conddist(f, k, l) * joint_deg_dist(k, l) for k, l in combinations[d-deg_low_lim*2][1]])
    assert abs(sum([sum([join_d_dist(f, d) for d in range(deg_low_lim*2, deg_high_lim*2)]) for f in range(feat_low_lim, feat_high_lim)]) - 1) < 0.01, "join_d_dist is not normalized"
    
    cond_d_dist = lambda f, d: sum([conddist(f, k, l) for k, l in combinations[d-deg_low_lim*2][1]])
    
    data = np.zeros((feat_high_lim, deg_high_lim*2))
    for f in range(feat_low_lim, feat_high_lim):
        for d in range(deg_low_lim*2, deg_high_lim):
            data[f, d] = cond_d_dist(f, d)

    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.imshow(data, cmap="inferno")
    plt.colorbar()
    plt.title("Conditional feature dist.")
    plt.xlabel("Degree sum k+l")
    plt.ylabel("Feature f")
    plt.savefig("./results/figures/PMA_POS.pdf", bbox_inches='tight')
    plt.clf()
    plt.close()

def criticalpoint(excdegdist, conddist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim):
    eig_vals = []
    for F0 in range(0, feat_high_lim+1):
        G = np.zeros((deg_high_lim, deg_high_lim))
        for i in range(deg_low_lim, deg_high_lim):
            for j in range(deg_low_lim, deg_high_lim):
                G[i, j] = (i)*excdegdist(i)*sum([conddist(f, (i+1), (j+1)) for f in range(0, F0)])
        eigenvalues, eigenvectors = eig(G[deg_low_lim:, deg_low_lim:])
        eig_vals.append(max(eigenvalues.real.tolist()))
    
    eig_vals = np.array(eig_vals)
    eig_vals[eig_vals > 1] = 0
    crit_val = np.argmax(eig_vals)
    return crit_val

def other_dist():
    beta = 1.0

    deg_low_lim = 0
    deg_high_lim = 80#20
    
    #upper_limit = 100#int(100000**0.5)+1
    feat_low_lim = 1
    feat_high_lim = 21
    bins = np.arange(0, feat_high_lim)

    degdist, excdegdist, joint_deg_dist, conddist, uncorr_conddist = generate_dist(beta, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    plot_conddist(conddist, joint_deg_dist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)

    """
    crit_point_corr = criticalpoint(excdegdist, conddist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    crit_point_uncorr = criticalpoint(excdegdist, uncorr_conddist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    print(f"Critical points: corr={crit_point_corr}, uncorr={crit_point_uncorr}")

    corr_sol = computeAnalitycalSolution(degdist, excdegdist, conddist, bins, deg_high_lim)
    print("corr sol", corr_sol)
    uncorr_sol = computeAnalitycalSolution(degdist, excdegdist, uncorr_conddist, bins, deg_high_lim)
    print("uncorr sol", uncorr_sol)
    
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.plot(bins,  corr_sol,    markersize=8, linestyle='dashed', color="k", marker='o', label="test. Corr. ")
    plt.plot(bins,  uncorr_sol,    markersize=8, linestyle='dashed', color="k", marker='s', label="test. UnCorr. ")
    """

    #plt.plot(bins, [0.0, 0.0, 0.7378219813049728, 0.9099138279974753, 0.9585618822375803, 0.977786002957832, 0.9869479728256523, 0.99186373072564, 0.9947234177131226, 0.996486178514336, 0.9976200980851169, 0.9983730290587156, 0.9988848384701282, 0.9992385892641166, 0.9994857151712755, 0.9996591918363473, 0.9997807902169156, 0.9998652477162439, 0.9999227572970889, 0.999960501198801, 0.9999836227765787],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Corr. β=0.1")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    #plt.plot(bins, [0.0, 0.0, 0.7407134938778882, 0.9146719219247015, 0.9617681058631726, 0.9798960293501946, 0.9883787493063113, 0.9928652628335835, 0.9954432871806593, 0.9970143859680095, 0.9980136876030142, 0.9986694835622577, 0.999109584987801, 0.9994093617674829, 0.9996151838996563, 0.9997566013399259, 0.9998530107541168, 0.9999174683460428, 0.9999589537872939, 0.9999837587194234, 0.9999963627474245],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Uncorr. β=0.1")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")    

    #plt.plot(bins,  [0.0, 0.0, 0.9258263190410174, 0.9875754179588777, 0.9964757394514941, 0.9986790377590967, 0.9994133606362603, 0.9997086329529812, 0.999843644687281, 0.9999113715657016, 0.9999477753229177, 0.9999683970806703, 0.9999805569294181, 0.9999879478580295, 0.9999925401644301, 0.9999954348550777, 0.9999972715056787, 0.9999984340331675, 0.999999159412906, 0.9999995974475399, 0.9999998447815518],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Corr. β=1.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins, [0.0, 0.0, 0.9305972488496421, 0.9890222782349253, 0.9969728967805673, 0.9988855892270565, 0.9995119093585009, 0.9997605653631343, 0.9998730833907751, 0.9999289964960455, 0.9999587725935394, 0.9999754761583032, 0.9999852195211392, 0.9999910676823973, 0.9999946465607943, 0.9999968596587829, 0.9999982288764181, 0.9999990656771314, 0.9999995611361576, 0.9999998352437157, 0.9999999648513292],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Uncorr. β=1.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins,[0.0, 0.0, 0.9999992498154265, 0.999999999899736, 0.9999999999998369, 0.9999999999999987, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998],    markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Corr. β=10.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    #plt.plot(bins, [0.0, 0.0, 0.9999995090919881, 0.999999999943108, 0.9999999999999136, 0.9999999999999992, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998, 0.9999999999999998],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Uncorr. β=10.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")



    plt.plot(bins, [0.0, 0.0, 0.9794271345105957, 0.9969203456234808, 0.9991640352669732, 0.9996936542989313, 0.9998657172019502, 0.9999338326177829, 0.9999646610429125, 0.9999800188708202, 0.9999882351183775, 0.9999928752527409, 0.9999956067153521, 0.9999972661236136, 0.9999982979259776, 0.9999989496498669, 0.9999993647090357, 0.9999996290181481, 0.9999997955304394, 0.9999998976863107, 0.9999999570473306],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Corr. α=-2")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    plt.plot(bins, [0.0, 0.0, 0.9817911897751269, 0.9974180259197138, 0.9993187722762328, 0.9997551128477591, 0.9998943207257188, 0.9999486764387817, 0.9999729906076865, 0.9999849711323494, 0.9999913101449922, 0.9999948480777032, 0.9999969033141967, 0.9999981327337659, 0.9999988829655442, 0.9999993457764218, 0.999999631520943, 0.9999998058418806, 0.9999999088936651, 0.9999999658276078, 0.9999999927154581],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Uncorr. α=-2")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")    

    plt.plot(bins,  [0.0, 0.0, 0.9258263190410174, 0.9875754179588777, 0.9964757394514941, 0.9986790377590967, 0.9994133606362603, 0.9997086329529812, 0.999843644687281, 0.9999113715657016, 0.9999477753229177, 0.9999683970806703, 0.9999805569294181, 0.9999879478580295, 0.9999925401644301, 0.9999954348550777, 0.9999972715056787, 0.9999984340331675, 0.999999159412906, 0.9999995974475399, 0.9999998447815518],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Corr. α=-3")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins, [0.0, 0.0, 0.9305972488496421, 0.9890222782349253, 0.9969728967805673, 0.9988855892270565, 0.9995119093585009, 0.9997605653631343, 0.9998730833907751, 0.9999289964960455, 0.9999587725935394, 0.9999754761583032, 0.9999852195211392, 0.9999910676823973, 0.9999946465607943, 0.9999968596587829, 0.9999982288764181, 0.9999990656771314, 0.9999995611361576, 0.9999998352437157, 0.9999999648513292],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Uncorr. α=-3")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    plt.plot(bins,[0.0, 0.0, 0.777341904442012, 0.9590283055136457, 0.9880170767848305, 0.9954465468185728, 0.9979643033150543, 0.99898602746069, 0.9994556623617696, 0.9996919148726726, 0.999819043473536, 0.9998910383589916, 0.9999334248811044, 0.9999591137403029, 0.9999750048394335, 0.9999849577763991, 0.9999912155859068, 0.9999951246025405, 0.9999975154142294, 0.9999989125478587, 0.9999996539008791], markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Corr. α=-4")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    plt.plot(bins, [0.0, 0.0, 0.7774389901337689, 0.9609966168280224, 0.9888514747235325, 0.9958207829873037, 0.9981497378302705, 0.9990858821578308, 0.9995130307473131, 0.9997265602775638, 0.9998407829107133, 0.9999050811868057, 0.9999426908726259, 0.9999653158688823, 0.9999791875024704, 0.9999877789126836, 0.9999931014490475, 0.9999963580900619, 0.9999982882425139, 0.9999993570174356, 0.9999998627597902],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Uncorr. α=-4")
    plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")
 
    plt.rc('legend', fontsize=12)#, fontsize=12)

    plt.xlim((-0.5, len(bins)-0.5))
    plt.ylim((-0.1, 1.1))
    #plt.title("Negative Feature-Degree Correlation \nER network ⟨k⟩=3")
    #plt.title("Negative Feature-Degree Correlation \nSF network α=-3")
    #plt.title("Negative Feature-Degree Correlation \nER network β=1.0")
    plt.title("Negative Feature-Degree Correlation \nSF network  β=1.0")
    plt.legend(loc='lower right')#, frameon=False, bbox_to_anchor=(1.0, 1))
    plt.xlabel(r'$F_0$')
    plt.ylabel("S")
    plt.xticks(list(range(0, len(bins)+1, 2)))
    plt.savefig("./results/figures/PLC_SF_NEG_DEG.pdf", bbox_inches='tight')

if __name__ == "__main__":
    #main()
    other_dist()
    #empirical_perc()