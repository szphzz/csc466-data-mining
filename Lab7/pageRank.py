import time
import sys
import pandas as pd
import numpy as np


def pageRank(df, d = 0.85, epsilon = 1e-6, max_i = 100):
    nodes = list(set(df.iloc[:,0].tolist() + df.iloc[:,2].tolist()))
    ranks = pd.DataFrame.from_dict(dict.fromkeys(nodes, 1/len(nodes)), orient='index')

    in_dict = {} # key is each node, value is list of nodes that point to key
    for node in nodes:
        in_dict[node] = df[df.iloc[:,0] == node].iloc[:,2].to_list()
    out_dict = df.iloc[:,2].value_counts().to_dict()
    read_time = time.time()
    
    i = 0
    times = []
    while i + 1 < max_i:
        beginning = time.time()
        if (i > 0) and (np.allclose(ranks.iloc[:,i].values, ranks.iloc[:,i - 1].values, atol=epsilon)):
            break
        col = []
        for node in nodes:
            updated = 0
            for link in in_dict[node]:
                updated += ranks.loc[link, i] / out_dict[link]
            col.append((1 - d) / len(nodes) + d * updated)
        col /= sum(col)
        i += 1
        ranks.insert(i, i, col)
        end = time.time()
        times.append(end - beginning)

    output = ranks.iloc[:,-1].sort_values(ascending=False)
    rank = 0
    for line in output:
        print(f"{rank + 1}  {output.index[rank]} with pagerank: {output[rank]}")
        rank += 1

    print(f"\nRead time: {read_time - t0} s")
    print(f"Average processing time: {sum(times) / len(times)} s")
    print(f"Total processing time: {sum(times)} s")
    print(f"Number of iterations: {i}")

    return output.index

if __name__ == '__main__':
    t0 = time.time()
    df = pd.read_csv(sys.argv[1], usecols=range(4), header=None)
    df = df.applymap(lambda x: x.replace('"', '').strip() if isinstance(x, str) else x)
    pageRank(df)

# python3 pageRank.py NCAA_football.csv
# python3 pageRank.py dolphins.csv
# python3 pageRank.py lesmis.csv