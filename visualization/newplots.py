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
        return sum([sum([excdegdist(l-1) * conddist(f, k, l) * (1-u[l]) for l in range(0, upper_limit)]) for f in range(1, F0)])

    def func(F0):
        def vecfunc(u):
            result = np.zeros_like(u)
            result[0] = 1
            for k in range(1, upper_limit):
                result[k] = (1-psi(u, k, F0))**(k-1)
            return u-result

        sol = optimize.root(vecfunc, np.zeros((upper_limit, 1))+0.0, method='lm')
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

    degdist = lambda k: binom.pmf(k, 100000, 0.00003)
    #C = sum([ks**(-3) for ks in range(lower_limit, upper_limit) if ks>=2])
    #degdist = lambda k: (k**(-3)/C) if k>=2 else 0

    degdistmean = sum([k*degdist(k) for k in range(deg_low_lim, deg_high_lim)])

    excdegdist = lambda k: degdist(k+1)*(k+1)/degdistmean
            
    joint_deg_dist = lambda k, l: excdegdist(k-1)*excdegdist(l-1) if deg_low_lim<=k<deg_high_lim and deg_low_lim<=l<deg_high_lim else 0.0
    assert abs(sum([sum([joint_deg_dist(k, l) for k in range(deg_low_lim, deg_high_lim)]) for l in range(deg_low_lim, deg_high_lim)]) - 1) < 0.01, "joint_deg_dist is not normalized"
    
    #_conddist = lambda f, k, l: (f+k+l)**(-3-beta)*zeta(3 + beta, 1 + k + l)**(-1)
    _conddist = lambda f, k, l: (k+l)**(2+beta) * zeta(2 + beta, (1 + k + l)/(k + l)) * (1+f*(k+l))**(-2-beta) if k+l>0 else 0.0
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
    beta = 5

    
    deg_low_lim = 0
    deg_high_lim = 20
    bins = np.arange(deg_low_lim, deg_high_lim)
    #upper_limit = 100#int(100000**0.5)+1
    feat_low_lim = 1
    feat_high_lim = 21

    degdist, excdegdist, joint_deg_dist, conddist, uncorr_conddist = generate_dist(beta, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    plot_conddist(conddist, joint_deg_dist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)

    #"""
    crit_point_corr = criticalpoint(excdegdist, conddist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    crit_point_uncorr = criticalpoint(excdegdist, uncorr_conddist, deg_low_lim, deg_high_lim, feat_low_lim, feat_high_lim)
    print(f"Critical points: corr={crit_point_corr}, uncorr={crit_point_uncorr}")

    corr_sol = computeAnalitycalSolution(degdist, excdegdist, conddist, bins, deg_high_lim)
    print("corr sol", corr_sol)
    uncorr_sol = computeAnalitycalSolution(degdist, excdegdist, uncorr_conddist, bins, deg_high_lim)
    print("uncorr sol", uncorr_sol)
    #"""

    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=16) 
    matplotlib.rc('ytick', labelsize=16) 
    plt.plot(bins,  corr_sol,    markersize=8, linestyle='dashed', color="k", marker='o', label="test. Corr. ")
    plt.plot(bins,  uncorr_sol,    markersize=8, linestyle='dashed', color="k", marker='s', label="test. UnCorr. ")

    #plt.plot(bins, [0.0, 0.0, 0.7353822892781817, 0.8387245842183323, 0.86802426744226, 0.8815384980017837, 0.8892436503306181, 0.8941973767404278, 0.89763907141934, 0.9001636909772223, 0.9020914880733355, 0.9036097756616, 0.9048352515301945, 0.9058443231702099, 0.9066890678330459, 0.9074061760664842, 0.9080222274157729, 0.9085569387276139, 0.909025238135516, 0.9094386312225633, 0.9098061256479215],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Corr. β=0.1")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    #plt.plot(bins, [0.0, 0.0, 0.7497197235088634, 0.8685515643298898, 0.8994860675120148, 0.9131548751994338, 0.9207485209597519, 0.925546452485181, 0.9288387025576573, 0.931231267355004, 0.9330450241374199, 0.9344652423276061, 0.9356061494199985, 0.9365418955848495, 0.9373226567795759, 0.9379835681243066, 0.9385499485435147, 0.9390404938497626, 0.9394693023105369, 0.9398472013302197, 0.9401826408392687],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Uncorr. β=0.1")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")    

    #plt.plot(bins,  [0.0, 0.0, 0.8301566209190937, 0.8705648503514883, 0.8807322599049875, 0.8848685292567107, 0.8869658669829111, 0.8881775107030287, 0.8889413088360331, 0.8894540095466467, 0.889814909466565, 0.8900785994108659, 0.8902771381722632, 0.8904303678066816, 0.8905511073356005, 0.8906479397809752, 0.8907267884066118, 0.890791848400869, 0.8908461586944022, 0.8908919645055035, 0.8909309538163134],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Corr. β=1.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins, [0.0, 0.0, 0.8791739317582663, 0.9215617142876549, 0.9312571133802496, 0.935060007154015, 0.9369527722552327, 0.9380342226073979, 0.9387110414741286, 0.9391630672321614, 0.939480077929429, 0.9397110455780755, 0.939884560547486, 0.9400182376938544, 0.9401234159738102, 0.9402076650556483, 0.9402761962056683, 0.940332693315572, 0.9403798188461365, 0.9404195551510648, 0.9404533435487503],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Uncorr. β=1.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins,[0.0, 0.0, 0.8732910077710266, 0.9310132363919857, 0.9380197657879639, 0.9396726209895918, 0.9401774616400018, 0.9403561734056725, 0.9404262696961887, 0.9404560167502165, 0.9404694731127852, 0.940475895293806, 0.9404791034521146, 0.9404807879356047, 0.9404816859404191, 0.9404826444936625, 0.9404829126307677, 0.9404830956836102, 0.9404934751576459, 0.9404935353031363, 0.9404832948543033],    markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Corr. β=10.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    #plt.plot(bins, [0.0, 0.0, 0.8474858141918857, 0.9199870665962194, 0.9335425150667935, 0.9376925108975804, 0.939242297209506, 0.9398906235247414, 0.9401839912515481, 0.940324997267607, 0.9403961883173413, 0.9404336728703854, 0.940454112501037, 0.9404656196660889, 0.9404727421447061, 0.9404766962827997, 0.9404790973153248, 0.9404805855604039, 0.9404815255213824, 0.9404821295184637, 0.9404928031229384],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Uncorr. β=10.0")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")



    #plt.plot(bins, [0.0, 0.0, 0.0, -1.368987532640987e-15, 0.4323670904453938, 0.5986548890128854, 0.6747696943881307, 0.7157974370915456, 0.7404139477676004, 0.7563354124509709, 0.7672176027642477, 0.7749757267574549, 0.7806932468297991, 0.7850212858768798, 0.7883706661661126, 0.7910111623923334, 0.7931259343993927, 0.7948428948660166, 0.7962535592355786, 0.7974247873932335, 0.7984063284046933],    markersize=8, linestyle='dashed', color="#d95f02", marker='o', label="Corr. ⟨k⟩=2")
    #plt.axvline(x=3, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")

    #plt.plot(bins, [0.0, 0.0, 0.0, 0.2486704542458534, 0.509090293203414, 0.6188107777298584, 0.67658243760671, 0.71113995720606, 0.7335918823767774, 0.749047879222844, 0.7601534035890459, 0.7684009306543489, 0.7746891802114463, 0.779587727562731, 0.783472719987293, 0.7866011170246303, 0.7891534993838355, 0.7912598604068988, 0.7930157372661654, 0.7944926525503575, 0.7957455282836652],       markersize=8, linestyle='dashed', color="#d95f02", marker='s', label="Uncorr. ⟨k⟩=2")
    #plt.axvline(x=2, ymin=-0.1, ymax=0.5, ls=":", color="#d95f02")    

    #plt.plot(bins,  [0.0, 0.0, 0.0, 0.5231459768427467, 0.7739477552042368, 0.8551222897681139, 0.8903377674485979, 0.9085802354148537, 0.9192148813787179, 0.9259522873717256, 0.9304894544823702, 0.9336898863233851, 0.9360306236204031, 0.9377929046099373, 0.9391514565417896, 0.9402196037173464, 0.9410735246082635, 0.9417659972241118, 0.9423345414430236, 0.9428064292592933, 0.9432018708677999],    markersize=8, linestyle='dashed', color="#1b9e77", marker='o', label="Corr. ⟨k⟩=3")
    #plt.axvline(x=2, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    #plt.plot(bins, [0.0, 0.0, 1.6653345360083642e-15, 0.5781921107305605, 0.7586848693872152, 0.8309396713174889, 0.8676243163504858, 0.8890161306655356, 0.902674276103231, 0.9119690945062623, 0.9186001445505841, 0.9235054062838153, 0.9272395704482836, 0.9301490544240533, 0.9324599644452408, 0.9343253485594228, 0.9358520093105062, 0.9371164316245626, 0.9381746202895921, 0.9390683826186882, 0.9398294591644714],       markersize=8, linestyle='dashed', color="#1b9e77", marker='s', label="Uncorr. ⟨k⟩=3")
    #plt.axvline(x=2, ymin=-0.1, ymax=0.5, ls=":", color="#1b9e77")

    #plt.plot(bins,[0.0, 0.0, 8.133452246554952e-18, 0.7196347510474154, 0.8839414740170546, 0.9338270997194981, 0.9542739140591093, 0.9643849935036394, 0.9700698246466196, 0.9735738754623053, 0.9758857074900895, 0.9774918100914842, 0.9786533643809248, 0.97952068220977, 0.9801852903547074, 0.9807055701768002, 0.9811202306351609, 0.9814557883913257, 0.9817309199302969, 0.9819590978201356, 0.982150244333648], markersize=8, linestyle='dashed', color="#7570b3", marker='o', label="Corr. ⟨k⟩=4")
    #plt.axvline(x=2, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")

    #plt.plot(bins, [0.0, 0.0, 0.08973427773678024, 0.7157901285124687, 0.8553720200961985, 0.9081588428961229, 0.9337310335916522, 0.9480871396311473, 0.9569786749083051, 0.9628852422657646, 0.9670194466010126, 0.9700322402413436, 0.9722991506379833, 0.9740495927138721, 0.9754304179868553, 0.9765393411295596, 0.9774435374916903, 0.9781904903930779, 0.9788145779754526, 0.9793412144599466, 0.9797895339437435],       markersize=8, linestyle='dashed', color="#7570b3", marker='s', label="Uncorr. ⟨k⟩=4")
    #plt.axvline(x=1, ymin=-0.1, ymax=0.5, ls=":", color="#7570b3")
 
    plt.rc('legend', fontsize=12)#, fontsize=12)

    plt.xlim((-0.5, len(bins)-0.5))
    plt.ylim((-0.1, 1.1))
    plt.title("Negative Feature-Degree Correlation \nER network ⟨k⟩=3")
    #plt.title("Negative Feature-Degree Correlation \nER network β=1.0")
    plt.legend(loc='lower right')#, frameon=False, bbox_to_anchor=(1.0, 1))
    plt.xlabel(r'$F_0$')
    plt.ylabel("S")
    plt.xticks(list(range(0, len(bins)+1, 2)))
    plt.savefig("./results/figures/PLC_ER_NEG.pdf", bbox_inches='tight')

if __name__ == "__main__":
    #main()
    other_dist()
    #empirical_perc()