import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import multiprocess as mp
from alive_progress import alive_bar

DATASET = "otc"
input_path = './data/bitcoin/bitcoin_{}.txt'.format(DATASET)
#input_path = './data/bitcoin/test_data.txt'
output_path = './data/btc_{}/'.format(DATASET)

df = pd.read_csv(input_path, sep=" ")



df = df.drop('time', axis=1)

df.loc[:, ['from', 'to']] -= 1 #node starts from 0
df.loc[:, ['feature']] += 10 #feature range: -10:10 -> 0:20

df = df.sort_values("to", kind='stable')
df = df.sort_values("from", kind='stable')

edges0 = df.shape[0]
df = df.drop(df[df[['from','to']].nunique(axis=1) == 1].index) #remove self edges
edges1 = df.shape[0]

print(df)

new_df = df.copy()
grouped_df = new_df.groupby(['from', 'to']).mean().reset_index()

merged_df = pd.merge(grouped_df, grouped_df.rename(columns={'from': 'to', 'to': 'from', 'feature': 'reverse_feature'}), on=['from', 'to'])
merged_df['average_feature'] = (merged_df['feature'] + merged_df['reverse_feature']) / 2
new_df = pd.merge(df, merged_df[['from', 'to', 'average_feature']], how="outer", on=['from', 'to'])
new_df.loc[~new_df["average_feature"].isna(), "feature"] = new_df["average_feature"]
new_df.drop("average_feature", axis=1, inplace=True)
new_df.drop_duplicates(subset=['from', 'to'], keep='first', inplace=True)
df = new_df

df = df.loc[pd.DataFrame(np.sort(df[['from','to']],1),index=df.index).drop_duplicates(keep='first').index] #remove multi edges
edges2 = df.shape[0]
print("Removed {} self edges and {} multi edges".format(edges0-edges1, edges1-edges2))

df["feature"] = df["feature"].astype(int)

df['from'], df['to'] = np.where(df['from']>df['to'], (df['to'], df['from']), (df['from'], df['to']))

df = df.sort_values("to", kind='stable')
df = df.sort_values("from", kind='stable')

counter = 0
while counter < max(df[["from", "to"]].max().to_list()):
    while counter not in df[["from", "to"]].values:
        df.loc[df["from"] > counter, "from"] -= 1
        df.loc[df["to"] > counter, "to"] -= 1
    counter += 1

print(df)
df.to_csv('./data/bitcoin/bitcoin_{}_processed.txt'.format(DATASET), sep=' ', index=False)


import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df'
column_name = 'feature'
column_data = df[column_name]

# Plotting the probability distribution
plt.hist(column_data, bins=20, density=True)  # Adjust the number of bins as needed
plt.xlabel(column_name)
plt.ylabel('Probability')
plt.title('Probability Distribution of {}'.format(column_name))
plt.savefig("./results/figures/FEATUREDIST.pdf", bbox_inches='tight')
plt.clf()
plt.close()


if __name__ == "__main__":
    pass