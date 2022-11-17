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

    print("i: {}, size: {}".format(i, df.shape))
    #df['feature'] = df['feature'].apply(lambda x: round(float(x)*10)+50)
    df['feature'] = df['feature'].apply(lambda x: float(x))
    df = df[df['feature'].between(-5, 5)]
    df['feature'] = df['feature'].apply(lambda x: round(x*10)+50)
    df["feature"].plot(kind="hist")
    plt.show()
    print("plotted")
    quit()

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

    if i!=11:
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

    func(params[11])

    #with alive_bar(len(params), theme='smooth') as bar:
    #    with mp.Pool(mp.cpu_count()) as pool:
    #        for __ in pool.imap_unordered(func, params):
    #            bar()