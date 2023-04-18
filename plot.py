import matplotlib.pyplot as plt
import pandas as pd


def plot_df(df, groupby, compare, X, Y):
    u = df[groupby].drop_duplicates()
    num_plots = len(u)
    num_rows = int(num_plots**0.6)
    num_cols = int(num_plots / num_rows) + 1

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 12, num_rows * 6),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten().tolist()

    for group_name, df1 in df.groupby(groupby):
        ax = axes.pop(0)
        for name, df2 in df1.groupby(compare):
            df3 = df2.sort_values(X)
            ax.plot(
                df3[X].to_numpy(), df3[Y].to_numpy(), label=[f"{name} {y}" for y in Y]
            )
        ax.set_title(group_name)
        ax.set_xlabel(X)
        ax.set_ylabel(Y)
        ax.legend()
        ax.grid()
    fig.show()


paths = {
    "LAPTOP RTX2080SMQ": "data/laptop-rtx2080smq-tongfang.csv",
    "LAPTOP 3080": "data/laptop-rtx3080-legion.csv",
    "DESKTOP 3090": "data/desktop-rtx3090.csv",
    "DESKTOP 3090 ZOTAC": "data/desktop-rtx3090-zotac.csv",
    "CLOUD V100": "data/cloud-v100.csv",
}

dfs = []
for k, v in paths.items():
    df = pd.read_csv(v)
    df["PC"] = k
    dfs.append(df)

full_df = pd.concat(dfs)
full_df = full_df.sort_values(by=["MODEL", "DS", "BS", "PC"])

plot_df(
    full_df,
    groupby=["DS", "MODEL"],
    compare=["PC"],
    X=["BS"],
    Y=[
        "TRAIN_SPE",  # "VALID_SPE",
    ],
)
