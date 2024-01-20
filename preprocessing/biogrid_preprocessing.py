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

class Biogrid(Preprocess):
    def __init__(self, path_to_raw: str, path_to_output: str, layoutfile) -> None:
        super().__init__(path_to_raw, path_to_output, ".edges")

        self.df_layout = pd.read_csv(layoutfile, sep=" ")

        self.na_nodes = self.df_layout["nodeID"].loc[self.df_layout["nodeSymbol"].isna()].to_list()
        

    def loadFiles(self):
        files_list = [f for f in glob.glob(self.path_to_raw+"*"+self.filetype)]
        with tqdm(total=len(files_list)) as pbar:
            for f in files_list:
                df = pd.read_csv(f, sep=" ", names=["from", "to", "feature"], header=None)
                df = df[df["feature"] != "-"]
                df = df[~df['from'].isin(self.na_nodes)]
                df = df[~df['to'].isin(self.na_nodes)]

                if(df.shape[0] > 100):
                    self.dataframes.append(df)

                pbar.update()

    def process_aggregation(self, min_f=-100, max_f=100, dx=0.1):
        with tqdm(total=9) as pbar:
            
            df = self.dataframes[0]
            for df_tmp in self.dataframes[1:]:
                df = df.append(df_tmp)
            pbar.update()
            
            df['feature'] = df['feature'].apply(lambda x: float(x))
            pbar.update()
            
            df.loc[df["from"]>df["to"], ['from', 'to']] = (df.loc[df["from"]>df["to"], ['to', 'from']].values)
            pbar.update()
            
            df['feature'] = df.groupby(['from', 'to'])['feature'].transform('sum')
            df.drop_duplicates(subset=['from', 'to'], inplace=True)
            pbar.update()
            
            df.drop(df[df["from"]==df["to"]].index, inplace=True) #remove self edges
            pbar.update()
            
            df.sort_values("to", kind='stable', inplace=True)
            df.sort_values("from", kind='stable', inplace=True)
            pbar.update()
            
            unique_nodes = pd.unique(df[['from', 'to']].values.ravel())
            node_map = {old: new for old, new in zip(sorted(unique_nodes), range(len(unique_nodes)))}
            df.replace({"from": node_map, "to": node_map}, inplace=True)
            pbar.update()
            
            df = df[df['feature'].between(min_f, max_f)]
            df['feature'] = df['feature'].apply(lambda x: round(x/dx)-round(min_f/dx))
            pbar.update()
                
            df.sort_values("to", kind='stable', inplace=True)
            df.sort_values("from", kind='stable', inplace=True)
            pbar.update()
            
            self.dataframes = [df]
    

def main():
    working_folder = "./data/biogrid/"
    input_path = working_folder+"raw/"
    output_path = working_folder+"processed/"
    layout_file = input_path+"Homo_sapiens_layout.txt"#"test_layout.txt"

    bg = Biogrid(input_path, output_path, layout_file) 

    bg.loadFiles()
    bg.process_aggregation()
    bg.writeFiles()


if __name__ == "__main__":
    main()
    print("All done")