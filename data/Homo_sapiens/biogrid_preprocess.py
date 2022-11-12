import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
from alive_progress import alive_bar
import glob


input_path = './data/Homo_sapiens/'
output_path = './data/biogrid/'

def func(params):
    i, file = params
    df = pd.read_csv(file, sep=" ", names=["from", "to", "feature"], header=None)

    df = df[df["feature"] != "-"]

    df=df.replace({"from": index_map})
    df=df.replace({"to": index_map})


    if i<10:
        df.to_csv(output_path + "chunck_0{}.txt".format(i), sep=' ', index=False, header=False)
    else:
        df.to_csv(output_path + "chunck_{}.txt".format(i), sep=' ', index=False, header=False)
    #print("chunk {}: up to {}; {} edges".format(i, dt, df.shape[0]))

if __name__ == "__main__":
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))

    df_layout = pd.read_csv(input_path+"Homo_sapiens_layout.txt", sep=" ")
    df_layout = df_layout.dropna().reset_index().reset_index().rename(columns={'index': 'old_index', 'level_0': 'new_index'})
    #print(df_layout)
    index_map = dict(zip(df_layout["old_index"], df_layout["new_index"]))

    edge_files = [f for f in glob.glob(input_path+"*.edges")]
    params = [(i, f) for i, f in enumerate(edge_files)]
    #print(edge_files)

    with alive_bar(len(params), theme='smooth') as bar:
        with mp.Pool(mp.cpu_count()) as pool:
            for __ in pool.imap_unordered(func, params):
                bar()