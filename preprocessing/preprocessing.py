"""
Base class for processing temporal network datasets
"""
import glob
import pandas as pd

class Preprocess:
    def __init__(self, path_to_raw: str, path_to_output: str, filetype=".edge") -> None:
        self.path_to_raw = path_to_raw
        self.path_to_output = path_to_output
        self.filetype = filetype
        
        self.dataframes = [] #list of pandas df

    def loadFiles(self):
        for f in [f for f in glob.glob(self.path_to_raw+"*"+self.filetype)]:
            self.dataframes.append(pd.read_csv(f, sep=" ", names=["from", "to", "feature"], header=None))

    def process(self):
        raise NotImplementedError

    def writeFiles(self):
        for idx, df in enumerate(self.dataframes):
            df.to_csv(self.path_to_output + f"{idx}.txt", sep=' ', index=False, header=False)