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

from preprocessing import Preprocess

class Bitcoin(Preprocess):
    def __init__(self, path_to_raw: str, path_to_output: str, num_windows=25) -> None:
        super().__init__(path_to_raw, path_to_output, ".csv")
        self.num_windows = num_windows

    def loadFiles(self, dataset="alpha"):
        file_name = glob.glob(self.path_to_raw+"*"+dataset+self.filetype)[0]

        df = pd.read_csv(file_name)
        df.sort_values("time", kind='stable', inplace=True)

        start_time = df.iloc[0]['time']
        stop_time = df.iloc[-1]['time']
        ranges = [start_time + i*(stop_time-start_time)/self.num_windows for i in range(1, self.num_windows+1)]
        prev_t = start_time

        with tqdm(total=self.num_windows) as pbar:
            for i, t in enumerate(ranges):
                self.dataframes.append(df[df['time'].between(prev_t, t)])
                prev_t = t
                pbar.update()

    def _func(self, idx):
        df = self.dataframes[idx]

        df.drop('time', axis=1, inplace=True)

        df.loc[:, ['from', 'to']] -= 1 #node starts from 0
        df.loc[:, ['feature']] += 10 #feature range: -10:10 -> 0:20
        
        df.sort_values("to", kind='stable', inplace=True)
        df.sort_values("from", kind='stable', inplace=True)

        df.loc[df["from"]>df["to"], ['from', 'to']] = (df.loc[df["from"]>df["to"], ['to', 'from']].values)

        df['feature'] = df.groupby(['from', 'to'])['feature'].transform(np.mean)
        df.drop_duplicates(subset=['from', 'to'], inplace=True)

        df.drop(df[df["from"]==df["to"]].index, inplace=True)

        df['feature'] = df['feature'].apply(lambda x: round(x))
        
        unique_nodes = pd.unique(df[['from', 'to']].values.ravel())
        node_map = {old: new for old, new in zip(sorted(unique_nodes), range(len(unique_nodes)))}
        df.replace({"from": node_map, "to": node_map}, inplace=True)

        df.sort_values("to", kind='stable', inplace=True)
        df.sort_values("from", kind='stable', inplace=True)

        return idx, df

    def process(self):
        with tqdm(total=self.num_windows) as pbar:
            params = [idx for idx in range(self.num_windows)]
            with mp.Pool(mp.cpu_count()) as pool:
                for idx, df in pool.imap_unordered(self._func, params):
                    self.dataframes[idx] = df
                    pbar.update()
    

def main():
    DATASET = "alpha"
    working_folder = "./data/bitcoin/"
    input_path = working_folder+"raw/"
    output_path = working_folder+"processed/"

    bc = Bitcoin(input_path, output_path) 

    bc.loadFiles()
    bc.process()
    bc.writeFiles()


if __name__ == "__main__":
    main()
    print("All done")