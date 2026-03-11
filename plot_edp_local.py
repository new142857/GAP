import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def normalize_method(method: str) -> str:
    method = str(method).strip().upper().replace("_", "-")
    alias = {
        "GAP-INF": r"GAP-$\infty$",
        "SAGE-INF": r"SAGE-$\infty$",
        "GAP-EDP": "GAP-EDP",
        "SAGE-EDP": "SAGE-EDP",
        "GAP-EDPX": "GAP-EDPX",
        "GAP-NDP": "GAP-NDP",
        "SAGE-NDP": "SAGE-NDP",
        "MLP": "MLP",
        "MLP-DP": "MLP-DP",
    }
    return alias.get(method, method)


def load_local_csvs(csv_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"在目录下没找到 csv: {csv_dir}")

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if df.empty:
                continue
            df["id"] = os.path.splitext(os.path.basename(fp))[0]
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 跳过文件 {fp}: {e}")

    if not dfs:
        raise ValueError("所有 csv 都读失败了，或内容为空。")

    df = pd.concat(dfs, ignore_index=True)

    # 兼容一些可能的列名写法
    rename_map = {
        "train_acc_mean": "train/acc_mean",
        "val_acc_mean": "val/acc_mean",
        "test_acc_mean": "test/acc_mean",
        "train_acc_ci": "train/acc_ci",
        "val_acc_ci": "val/acc_ci",
        "test_acc_ci": "test/acc_ci",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "method" not in df.columns:
        raise KeyError("缺少 method 列")
    if "dataset" not in df.columns:
        raise KeyError("缺少 dataset 列")
    if "epsilon" not in df.columns:
        raise KeyError("缺少 epsilon 列")
    if "test/acc_mean" not in df.columns:
        raise KeyError("缺少 test/acc_mean 列")

    if "val/acc_mean" not in df.columns:
        print("[WARN] 没有 val/acc_mean，自动用 test/acc_mean 代替做最佳模型选择。")
        df["val/acc_mean"] = df["test/acc_mean"]

    df["method"] = df["method"].apply(normalize_method)

    return df


def plot_edge(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=2.0,
        rc={
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "lines.linewidth": 3,
            "lines.markersize": 10,
        },
    )

    cols = ["id", "method", "dataset", "epsilon", "val/acc_mean", "test/acc_mean"]
    methods = [r"GAP-$\infty$", "MLP", "GAP-EDP", "GAP-EDPX"]
    datasets = ["facebook", "reddit", "amazon"]
    epsilons = [0.1, 0.2, 0.5, 1, 2, 4, 8]

    df = df.copy()
    df = df[[c for c in cols if c in df.columns]]

    df.loc[df["method"] == "MLP", "epsilon"] = 0
    df.loc[df["method"] == r"GAP-$\infty$", "epsilon"] = np.inf

    df = df[df["method"].isin(methods)]
    df = df[df["dataset"].isin(datasets)]
    df = df[df["epsilon"].isin(epsilons + [0, np.inf])]

    if df.empty:
        raise ValueError("过滤后没有数据。请检查 method / dataset / epsilon 是否对上。")

    # 每个(dataset, method, epsilon)只保留 val 最好的那条
    idx = df.groupby(["dataset", "method", "epsilon"])["val/acc_mean"].idxmax().values
    df = df.loc[idx].reset_index(drop=True)

    inf_df = df[df["method"] == r"GAP-$\infty$"][["dataset", "test/acc_mean"]].drop_duplicates("dataset")
    mlp_df = df[df["method"] == "MLP"][["dataset", "test/acc_mean"]].drop_duplicates("dataset")

    inf_acc = inf_df.set_index("dataset")["test/acc_mean"].to_dict()
    mlp_acc = mlp_df.set_index("dataset")["test/acc_mean"].to_dict()

    edp_methods = ["GAP-EDP", "GAP-EDPX"]
    plot_df = df[df["method"].isin(edp_methods)].copy()

    if plot_df.empty:
        raise ValueError("没有找到 GAP-EDP / GAP-EDPX 的结果。请检查 method 字段。")

    g = sns.relplot(
        kind="line",
        data=plot_df,
        x="epsilon",
        y="test/acc_mean",
        hue="method",
        col="dataset",
        aspect=1.2,
        markers=["o", "d"],
        dashes=False,
        style="method",
        hue_order=edp_methods,
        col_order=datasets,
        style_order=edp_methods,
        facet_kws={"sharey": False, "sharex": False},
    )

    g.set(
        ylabel=None,
        xlabel=r"Privacy Cost $(\epsilon)$",
        xscale="log",
        xticks=epsilons,
        xlim=(epsilons[0] / 2, epsilons[-1] * 2),
    )

    palette = sns.color_palette("deep")

    for dataset, ax in zip(datasets, g.axes[0]):
        if dataset in inf_acc:
            ax.axhline(
                inf_acc[dataset],
                linestyle="dotted",
                label=r"GAP-$\infty$",
                color=palette[2],
                xmin=0.12,
                xmax=0.88,
            )
        if dataset in mlp_acc:
            ax.axhline(
                mlp_acc[dataset],
                linestyle="dashed",
                label="MLP",
                color=palette[3],
                xmin=0.12,
                xmax=0.88,
            )

        ax.set_title(dataset.capitalize(), fontdict={"fontweight": "bold"})
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.minorticks_off()

    handles, labels = g.axes[0][0].get_legend_handles_labels()
    legend_data = {label: handle for label, handle in zip(labels, handles)}
    g.legend.remove()

    legend_order = [m for m in methods if m in legend_data]
    if legend_order:
        g.axes[0][1].legend(
            loc="upper center",
            ncol=len(legend_order),
            bbox_to_anchor=(0.45, 1.35),
            handles=[legend_data[m] for m in legend_order],
            labels=legend_order,
        )

    g.axes[0][0].set_ylabel("Accuracy (%)")
    g.fig.subplots_adjust(hspace=0.4, wspace=0.2)
    g.fig.set_figwidth(16)

    pdf_path = os.path.join(out_dir, "edge_gap_edpx.pdf")
    png_path = os.path.join(out_dir, "edge_gap_edpx.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="./output/csv/GAP")
    parser.add_argument("--out_dir", type=str, default="./figs")
    args = parser.parse_args()

    df = load_local_csvs(args.csv_dir)

    print("\n[INFO] 读到的列：")
    print(sorted(df.columns.tolist()))

    print("\n[INFO] method 分布：")
    print(df["method"].value_counts(dropna=False))

    plot_edge(df, args.out_dir)


if __name__ == "__main__":
    main()