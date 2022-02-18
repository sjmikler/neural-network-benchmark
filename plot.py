import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df, groupby, compare, X, Y):
    for group_name, df1 in df.groupby(groupby):
        plt.figure(figsize=(12, 6))
        for name, df2 in df1.groupby(compare):
            df3 = df2.sort_values(X)
            plt.plot(df3[X], df3[Y], label=f"{name}")
        plt.title(group_name)
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.legend()
        plt.grid()
        plt.show()


paths = {
    "TF-QUIET": "tongfang-quiet.csv",
    "TF-PERF": "tongfang-performance.csv",
}

dfs = []
for k, v in paths.items():
    df = pd.read_csv(v)
    df["PC"] = k
    dfs.append(df)

full_df = pd.concat(dfs)
plot_df(
    full_df,
    groupby=["DS", "MODEL"],
    compare=["PC"],
    X="BS",
    Y=["TRAIN_SPE", "VALID_SPE"],
)
