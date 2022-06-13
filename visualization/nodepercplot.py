import csv
import numpy as np
import matplotlib.pyplot as plt

import yaml

def main():
    with open("./experiments/config.yaml") as file:
        config_params = yaml.load(file, Loader=yaml.FullLoader)
        #print(config_params)

    for exp_n, exp_params in config_params['giant_component'].items():
        with open("./results/raw/node_perc_giant_cluster_exp_{}.csv".format(exp_n)) as file:
            #header = next(csv.reader(file))
            row = next(csv.reader(file))

        row = [float(i)/int(exp_params['network_size']) for i in row] #convert to float and normalize
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
        elif exp_params['network_type'] == 'p':
            legend = "n={}, runs={}; degree dist.: p({}), xmax={}".format(exp_params['network_size'],
                                                                                    exp_params['runs'],
                                                                                    exp_params['param1'],
                                                                                    exp_params['param2'])


        plt.plot(np.arange(0, 1, 1/len(row)), row, label=legend)
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