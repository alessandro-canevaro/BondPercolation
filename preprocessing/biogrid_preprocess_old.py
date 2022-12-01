import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
from alive_progress import alive_bar
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


input_path = './data/Homo_sapiens/'
output_path = './data/biogrid/'
min_f, max_f = -100, 100
dx = 0.1

def func(params):
    i, file = params
    df = pd.read_csv(file, sep=" ", names=["from", "to", "feature"], header=None)

    df = df[df["feature"] != "-"]

    #print("i: {}, size: {}".format(i, df.shape))
    #df['feature'] = df['feature'].apply(lambda x: round(float(x)*10)+50)
    df['feature'] = df['feature'].apply(lambda x: float(x))
    df = df[df['feature'].between(min_f, max_f)]
    #min_f, max_f = df['feature'].min(), df['feature'].max()
    #print(i, round(min_f*10), round(max_f*10))

    #print("i: {}, size: {}".format(i, df.shape))
    df['feature'] = df['feature'].apply(lambda x: round(x/dx)-round(min_f/dx))
    #df["feature"].plot(kind="hist")
    #plt.show()
    #print("plotted")
    #quit()

    df=df.replace({"from": index_map})
    df=df.replace({"to": index_map})

    df = df.sort_values("to", kind='stable')
    df = df.sort_values("from", kind='stable')

    edges0 = df.shape[0]
    df = df.drop(df[df[['from','to']].nunique(axis=1) == 1].index) #remove self edges
    edges1 = df.shape[0]
    df = df.loc[pd.DataFrame(np.sort(df[['from','to']],1),index=df.index).drop_duplicates(keep='first').index] #remove multi edges
    edges2 = df.shape[0]
    #print("Removed {} self edges and {} multi edges".format(edges0-edges1, edges1-edges2))

    df['from'], df['to'] = np.where(df['from']>df['to'], (df['to'], df['from']), (df['from'], df['to']))

    df = df.sort_values("to", kind='stable')
    df = df.sort_values("from", kind='stable')

    counter = 0
    while(counter < max(df[["from", "to"]].max().to_list())):
        while(counter not in df[["from", "to"]].values):
            df["from"].loc[df["from"] > counter] -= 1
            df["to"].loc[df["to"] > counter] -= 1
        counter += 1


    if i<10:
        df.to_csv(output_path + "chunck_0{}.txt".format(i), sep=' ', index=False, header=False)
    else:
        df.to_csv(output_path + "chunck_{}.txt".format(i), sep=' ', index=False, header=False)
    #print("chunk {}: up to {}; {} edges".format(i, dt, df.shape[0]))

def aggregated_net(files):
    df = pd.read_csv(files[0], sep=" ", names=["from", "to", "feature"], header=None)
    df = df[df["feature"] != "-"]
    for file in files[1:]:
        df_tmp = pd.read_csv(file, sep=" ", names=["from", "to", "feature"], header=None)
        df_tmp = df_tmp[df_tmp["feature"] != "-"]
        df = df.append(df_tmp)

    df['feature'] = df['feature'].apply(lambda x: float(x))

    df=df.replace({"from": index_map})
    df=df.replace({"to": index_map})

    df.loc[df["from"]>df["to"], ['from', 'to']] = (df.loc[df["from"]>df["to"], ['to', 'from']].values)

    df['feature'] = df.groupby(['from', 'to'])['feature'].transform('sum')
    df = df.drop_duplicates(subset=['from', 'to'])

    df = df.drop(df[df["from"]==df["to"]].index) #remove self edges

    df = df.sort_values("to", kind='stable')
    df = df.sort_values("from", kind='stable')

    counter = 0
    with tqdm(total=15000) as pbar:
        while(counter < max(df[["from", "to"]].max().to_list())):
            while(counter not in df[["from", "to"]].values):
                df["from"].loc[df["from"] > counter] -= 1
                df["to"].loc[df["to"] > counter] -= 1
            counter += 1
            pbar.update()

    df = df[df['feature'].between(min_f, max_f)]

    df['feature'] = df['feature'].apply(lambda x: round(x/dx)-round(min_f/dx))

    df = df.sort_values("to", kind='stable')
    df = df.sort_values("from", kind='stable')


    df.to_csv(output_path + "aggregation.txt", sep=' ', index=False, header=False)
    print("written")

if __name__ == "__main__":
    print("started")
    #import cpuinfo
    #cpu = cpuinfo.get_cpu_info()
    #print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))

    df_layout = pd.read_csv(input_path+"Homo_sapiens_layout.txt", sep=" ")
    df_layout = df_layout.dropna().reset_index().reset_index().rename(columns={'index': 'old_index', 'level_0': 'new_index'})
    #print(df_layout)
    index_map = dict(zip(df_layout["old_index"], df_layout["new_index"]))

    edge_files = []
    for f in [f for f in glob.glob(input_path+"*.edges")]:
        df = pd.read_csv(f, sep=" ", names=["from", "to", "feature"], header=None)
        df = df[df["feature"] != "-"]
        if(df.shape[0] > 100):
            edge_files.append(f)
    #print(edge_files)

    aggregated_net(edge_files)
    #aggregated_net([input_path+"test.edges"])
    quit()

    params = [(i, f) for i, f in enumerate(edge_files)]
    #func(params[11])

    with alive_bar(len(params), theme='smooth') as bar:
        with mp.Pool(mp.cpu_count()) as pool:
            for __ in pool.imap_unordered(func, params):
                bar()