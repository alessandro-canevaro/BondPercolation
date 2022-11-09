import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
from alive_progress import alive_bar

input_path = './data/Rural_Malawi.csv'#'./data/Rural_Malawi.csv'
output_path = './data/rural_malawi/'#'./data/rural_malawi/'

df = pd.read_csv(input_path, sep=',')
#print(df)
#df = df.drop("idx", axis=1)
#df = df.drop("day", axis=1)
df = df.sort_values("time", kind='stable')

num_files = 25
start_time = df.iloc[0]['time']
stop_time = df.iloc[-1]['time']
ranges = [start_time + i*(stop_time-start_time)/num_files for i in range(1, num_files+1)]
#print("start {}, stop {}, ranges {}".format(start_time, stop_time, ranges))
#print(df)

prev_t = start_time
data_frames = []
for i, t in enumerate(ranges):
    data_frames.append((i, t, df[df['time'].between(prev_t, t)]))
    prev_t = t

def func(params):
    i, t, df = params

    df = df.drop('time', axis=1)

    df.loc[:, ['from', 'to']] -= 1 #node starts from 0

    count_series = df.groupby(['from', 'to']).size()
    df = count_series.to_frame(name = 'feature').reset_index()
    #print(df)
        
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

    #ax = df['feature'].plot.hist()
    #plt.show()

    dt = datetime.datetime.fromtimestamp(t).strftime('%Y_%m_%d')

    if i<10:
        df.to_csv(output_path + "chunck_0{}.txt".format(i), sep=' ', index=False, header=False)
    else:
        df.to_csv(output_path + "chunck_{}.txt".format(i), sep=' ', index=False, header=False)
    #print("chunk {}: up to {}; {} edges".format(i, dt, df.shape[0]))

if __name__ == "__main__":
    import cpuinfo
    cpu = cpuinfo.get_cpu_info()
    print('{}, {} cores'.format(cpu['brand_raw'], cpu['count']))
    with alive_bar(len(data_frames), theme='smooth') as bar:
        with mp.Pool(mp.cpu_count()) as pool:
            for __ in pool.imap_unordered(func, data_frames):
                bar()