import os
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def normalize_method(method: str) -> str:
    method = str(method).strip().upper().replace("_", "-")
    alias = {
        "GAP-INF": r"GAP-$\infty$",
        "SAGE-INF": r"SAGE-$\infty$",
        "GAP-EDP": "GAP-EDP",
        "SAGE-EDP": "SAGE-EDP",
        "GAP-EDPX": "GAP-EDPX",
        "GAP-EDPE": "GAP-EDPE",
        "GAP-EDPES": "GAP-EDPES",
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
    if "test/acc_mean" not in df.columns:
        raise KeyError("缺少 test/acc_mean 列")

    if "val/acc_mean" not in df.columns:
        print("[WARN] 没有 val/acc_mean，自动用 test/acc_mean 代替做最佳模型选择。")
        df["val/acc_mean"] = df["test/acc_mean"]

    df["method"] = df["method"].apply(normalize_method)
    return df


def infer_lambda_col(df: pd.DataFrame, user_specified: str = "") -> str:
    if user_specified:
        if user_specified not in df.columns:
            raise KeyError(f"--lambda_col={user_specified} 不在列中，现有列：{sorted(df.columns.tolist())}")
        return user_specified

    candidates = [
        "hop_noise_lambda",
        "lambda",
        "lamda",
        "schedule_lambda",
        "noise_lambda",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        "没找到 lambda 列。可检查是否存在下列之一："
        f"{candidates}；或者手动传 --lambda_col 指定。"
    )


def parse_dataset_list(raw: str, df: pd.DataFrame):
    if raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    default_order = ["facebook", "reddit", "amazon"]
    existing = df["dataset"].dropna().astype(str).unique().tolist()
    ordered = [d for d in default_order if d in existing]
    others = [d for d in existing if d not in ordered]
    return ordered + sorted(others)


def maybe_filter_epsilon(df: pd.DataFrame, epsilon):
    if epsilon is None:
        return df

    if "epsilon" not in df.columns:
        print("[WARN] 指定了 --epsilon，但数据里没有 epsilon 列，跳过该过滤。")
        return df

    # 非私有上限基线始终保留
    non_private_methods = {r"GAP-$\infty$", r"SAGE-$\infty$"}

    x = pd.to_numeric(df["epsilon"], errors="coerce")
    keep_eps = np.isclose(x, float(epsilon), atol=1e-12, rtol=0)
    keep_non_private = df["method"].isin(non_private_methods)

    return df[keep_eps | keep_non_private].copy()


def maybe_filter_hops(df: pd.DataFrame, hops):
    """根据hops参数过滤DataFrame"""
    if hops is None:
        return df

    if "hops" not in df.columns:
        print("[WARN] 指定了 --hops，但数据里没有 hops 列，跳过该过滤。")
        return df

    # 过滤指定hops的数据
    x = pd.to_numeric(df["hops"], errors="coerce")
    keep_hops = np.isclose(x, float(hops), atol=1e-12, rtol=0)

    return df[keep_hops].copy()


def best_by_group(df: pd.DataFrame, group_cols):
    idx = df.groupby(group_cols)["val/acc_mean"].idxmax().values
    return df.loc[idx].reset_index(drop=True)

def infer_strict_match_cols(df: pd.DataFrame, lambda_col: str):
    """
    推断“同一实验条件”要匹配的列。
    会排除：
    1) 指标列
    2) 画图专用/派生列
    3) 方法特有列（lambda / hop schedule）
    4) 路径、日志、设备等非核心实验条件
    """
    exclude_cols = {
        # 标识/元数据
        "id", "method", "dataset",
        "project", "output_dir", "logger", "device", "data_dir",

        # 指标列
        "train/acc_mean", "train/acc_std", "train/acc_ci",
        "val/acc_mean", "val/acc_std", "val/acc_ci",
        "test/acc_mean", "test/acc_std", "test/acc_ci",
        "train/loss_mean", "train/loss_std", "train/loss_ci",
        "val/loss_mean", "val/loss_std", "val/loss_ci",
        "duration_mean", "duration_std", "duration_ci",
        "epoch_mean", "epoch_std", "epoch_ci",

        # 噪声统计
        "noise_scale_mean", "noise_scale_std", "noise_scale_ci",

        # schedule 专属列：做“同家族曲线”时不应该拿来锁死
        lambda_col,
        "hop_noise_lambda",
        "hop_noise_mode",
        "hop_noise_factors",
        "hop_budget_weights",

        # 一般不作为聚合结果匹配键
        "seed",
    }

    return [c for c in df.columns if c not in exclude_cols]


def _series_match_value(series: pd.Series, value):
    """按数值/字符串稳健匹配单列"""
    if pd.isna(value):
        return series.isna()

    # 先尝试按数值匹配
    s_num = pd.to_numeric(series, errors="coerce")
    try:
        v_num = float(value)
        if s_num.notna().any():
            return np.isclose(s_num, v_num, atol=1e-12, rtol=0)
    except Exception:
        pass

    # 再按字符串匹配
    return series.astype(str).str.strip().str.lower() == str(value).strip().lower()


def filter_by_conditions(df: pd.DataFrame, cond: dict, match_cols):
    out = df.copy()
    for col in match_cols:
        if col not in out.columns or col not in cond:
            continue
        out = out[_series_match_value(out[col], cond[col])]
    return out.copy()


def pick_reference_row(df: pd.DataFrame, dataset: str, method: str, lambda_col: str, ref_lambda: float):
    """
    在指定 dataset/method 下，找到 lambda=ref_lambda 的参考记录；
    若有多条，取 val/acc_mean 最大的那条。
    """
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)].copy()
    if sub.empty:
        return None

    sub[lambda_col] = pd.to_numeric(sub[lambda_col], errors="coerce")
    sub = sub[sub[lambda_col].notna()].copy()
    sub = sub[np.isclose(sub[lambda_col], float(ref_lambda), atol=1e-12, rtol=0)].copy()

    if sub.empty:
        return None

    return sub.sort_values("val/acc_mean", ascending=False).iloc[0]


def build_strict_family_data(
    df: pd.DataFrame,
    datasets,
    curve_method: str,
    baseline_methods,
    lambda_col: str,
    ref_lambda: float,
):
    """
    严格同参：
    1) 先拿每个 dataset 上 curve_method 在 lambda=ref_lambda 的最佳记录做参考配置
    2) 再用这套配置筛整条 curve_method 曲线
    3) 再用这套配置分别筛 baseline_methods 横线
    """
    match_cols = infer_strict_match_cols(df, lambda_col)

    curve_parts = []
    baseline_maps = {m: {} for m in baseline_methods}
    ref_rows = {}

    for dataset in datasets:
        ref_row = pick_reference_row(
            df=df,
            dataset=dataset,
            method=curve_method,
            lambda_col=lambda_col,
            ref_lambda=ref_lambda,
        )

        if ref_row is None:
            print(f"[WARN] dataset={dataset} 没找到 {curve_method} 在 lambda={ref_lambda} 的参考记录，跳过。")
            continue

        ref_rows[dataset] = ref_row
        cond = {c: ref_row[c] for c in match_cols if c in ref_row.index}

        # 1) 严格同参的 EDPES 曲线
        curve_sub = df[(df["dataset"] == dataset) & (df["method"] == curve_method)].copy()
        curve_sub = filter_by_conditions(curve_sub, cond, match_cols)
        curve_sub[lambda_col] = pd.to_numeric(curve_sub[lambda_col], errors="coerce")
        curve_sub = curve_sub[curve_sub[lambda_col].notna()].copy()

        if not curve_sub.empty:
            curve_sub = best_by_group(curve_sub, ["dataset", "method", lambda_col])
            curve_parts.append(curve_sub)
        else:
            print(f"[WARN] dataset={dataset} 的 {curve_method} 严格同参曲线为空。")

        # 2) 严格同参的各 baseline 横线
        for base_method in baseline_methods:
            base_sub = df[(df["dataset"] == dataset) & (df["method"] == base_method)].copy()

            # GAP-EDP：严格同参
            if base_method == "GAP-EDP":
                base_sub = filter_by_conditions(base_sub, cond, match_cols)

            # GAP-∞：只要求 hops 一致（dataset 已经在上面筛了）
            elif base_method == r"GAP-$\infty$":
                if "hops" in base_sub.columns and "hops" in ref_row.index:
                    base_sub["hops"] = pd.to_numeric(base_sub["hops"], errors="coerce")
                    ref_hops = pd.to_numeric(pd.Series([ref_row["hops"]]), errors="coerce").iloc[0]
                    base_sub = base_sub[base_sub["hops"].notna()]
                    base_sub = base_sub[np.isclose(base_sub["hops"], float(ref_hops), atol=1e-12, rtol=0)].copy()

            if base_sub.empty:
                print(f"[WARN] dataset={dataset} 没找到可匹配的 {base_method}。")
                continue

            base_best = base_sub.sort_values("val/acc_mean", ascending=False).iloc[0]
            baseline_maps[base_method][dataset] = base_best["test/acc_mean"]

    if not curve_parts:
        raise ValueError("严格同参模式下没有构造出任何有效曲线，请检查 CSV 中是否真的有对应配置。")

    curve_df = pd.concat(curve_parts, ignore_index=True)
    return curve_df, baseline_maps, ref_rows, match_cols

def get_best_baseline(df: pd.DataFrame, method_name: str, value_col: str):
    sub = df[df["method"] == method_name].copy()
    if sub.empty:
        return {}
    sub = best_by_group(sub, ["dataset", "method"])
    return sub.set_index("dataset")[value_col].to_dict()


def prepare_curve_df(df: pd.DataFrame, datasets, curve_methods, lambda_col):
    curve_df = df[df["method"].isin(curve_methods)].copy()
    curve_df = curve_df[curve_df["dataset"].isin(datasets)].copy()
    curve_df[lambda_col] = pd.to_numeric(curve_df[lambda_col], errors="coerce")
    curve_df = curve_df[curve_df[lambda_col].notna()].copy()

    if curve_df.empty:
        raise ValueError(f"没有找到 {curve_methods} 的有效 lambda 数据。")

    curve_df = best_by_group(curve_df, ["dataset", "method", lambda_col])
    return curve_df


def plot_accuracy_vs_lambda(df: pd.DataFrame, out_dir: str, lambda_col: str, epsilon=None, datasets_raw: str = "",
                            hops=None, ref_lambda: float = 0.0):
    os.makedirs(out_dir, exist_ok=True)

    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.8,
        rc={
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "lines.linewidth": 3,
            "lines.markersize": 8,
        },
    )

    df = df.copy()
    if epsilon is not None:
        df = maybe_filter_epsilon(df, epsilon)
    if hops is not None:
        df = maybe_filter_hops(df, hops)

    datasets = parse_dataset_list(datasets_raw, df)

    keep_methods = ["GAP-EDPES", "GAP-EDP", r"GAP-$\infty$"]
    df = df[df["method"].isin(keep_methods)].copy()
    df = df[df["dataset"].isin(datasets)].copy()

    curve_df, baseline_maps, ref_rows, match_cols = build_strict_family_data(
        df=df,
        datasets=datasets,
        curve_method="GAP-EDPES",
        baseline_methods=["GAP-EDP", r"GAP-$\infty$"],
        lambda_col=lambda_col,
        ref_lambda=ref_lambda,
    )

    edp_base = baseline_maps.get("GAP-EDP", {})
    inf_base = baseline_maps.get(r"GAP-$\infty$", {})

    datasets = [d for d in datasets if d in curve_df["dataset"].unique()]
    xticks = sorted(curve_df[lambda_col].dropna().unique().tolist())

    print("\n[INFO] 严格同参匹配列：")
    print(match_cols)

    print("\n[INFO] 每个数据集使用的参考配置（来自 GAP-EDPES, lambda=0）：")
    for d in datasets:
        if d in ref_rows:
            ref = ref_rows[d]
            show_cols = [c for c in [
                "dataset", "epsilon", "hops", "encoder_layers", "base_layers", "head_layers",
                "combine", "hidden_dim", "activation", "optimizer", "learning_rate",
                "batch_norm", "epochs", "batch_size", "encoder_epochs"
            ] if c in ref.index]
            print(f"  - {d}:")
            print(ref[show_cols].to_dict())

    g = sns.relplot(
        kind="line",
        data=curve_df,
        x=lambda_col,
        y="test/acc_mean",
        hue="method",
        style="method",
        col="dataset",
        markers=True,
        dashes=False,
        hue_order=["GAP-EDPES"],
        style_order=["GAP-EDPES"],
        col_order=datasets,
        aspect=1.25,
        facet_kws={"sharey": False, "sharex": True},
    )

    g.set_axis_labels(r"Schedule parameter $(\lambda)$", None)
    palette = sns.color_palette("deep")

    for ax, dataset in zip(g.axes[0], datasets):
        if dataset in edp_base:
            ax.axhline(
                edp_base[dataset],
                linestyle="--",
                color=palette[2],
                label="GAP-EDP",
                xmin=0.03,
                xmax=0.97,
            )
        if dataset in inf_base:
            ax.axhline(
                inf_base[dataset],
                linestyle=":",
                color=palette[3],
                label=r"GAP-$\infty$",
                xmin=0.03,
                xmax=0.97,
            )

        ax.set_title(dataset.capitalize(), fontdict={"fontweight": "bold"})
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", rotation=45)

    handles, labels = g.axes[0][0].get_legend_handles_labels()
    legend_data = {label: handle for label, handle in zip(labels, handles)}
    if g.legend is not None:
        g.legend.remove()

    legend_order = [m for m in ["GAP-EDPES", "GAP-EDP", r"GAP-$\infty$"] if m in legend_data]
    if legend_order:
        g.axes[0][min(1, len(datasets) - 1)].legend(
            loc="upper center",
            ncol=len(legend_order),
            bbox_to_anchor=(0.5, 1.28),
            handles=[legend_data[m] for m in legend_order],
            labels=legend_order,
            frameon=True,
        )

    g.axes[0][0].set_ylabel("Accuracy (%)")
    g.fig.subplots_adjust(hspace=0.35, wspace=0.2)
    g.fig.set_figwidth(max(6 * len(datasets), 14))

    eps_tag = "all" if epsilon is None else str(epsilon).replace(".", "p")
    hops_tag = "all" if hops is None else str(hops).replace(".", "p")
    ref_tag = str(ref_lambda).replace(".", "p").replace("-", "m")

    pdf_path = os.path.join(out_dir, f"edge_gap_acc_vs_lambda_strict_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.pdf")
    png_path = os.path.join(out_dir, f"edge_gap_acc_vs_lambda_strict_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def plot_noise_scale_vs_lambda(df: pd.DataFrame, out_dir: str, lambda_col: str, epsilon=None, datasets_raw: str = "",
                               hops=None, ref_lambda: float = 0.0):
    if "noise_scale_mean" not in df.columns:
        raise KeyError("当前 CSV 没有 noise_scale 列，请先把训练时的 self.noise_scale 写入结果 CSV。")

    os.makedirs(out_dir, exist_ok=True)

    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.8,
        rc={
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "lines.linewidth": 3,
            "lines.markersize": 8,
        },
    )

    df = df.copy()
    if epsilon is not None:
        df = maybe_filter_epsilon(df, epsilon)
    if hops is not None:
        df = maybe_filter_hops(df, hops)

    datasets = parse_dataset_list(datasets_raw, df)

    # 先用严格同参逻辑构造出 EDPES 家族
    tmp_df = df[df["method"].isin(["GAP-EDPES", "GAP-EDP"])].copy()
    tmp_df = tmp_df[tmp_df["dataset"].isin(datasets)].copy()

    curve_df, _, _, _ = build_strict_family_data(
        df=tmp_df,
        datasets=datasets,
        curve_method="GAP-EDPES",
        baseline_methods=["GAP-EDP", r"GAP-$\infty$"],
        lambda_col=lambda_col,
        ref_lambda=ref_lambda,
    )

    curve_df["noise_scale"] = pd.to_numeric(curve_df["noise_scale_mean"], errors="coerce")
    curve_df = curve_df[curve_df["noise_scale"].notna()].copy()

    if curve_df.empty:
        raise ValueError("严格同参模式下没有可用于绘制 noise_scale vs lambda 的数据。")

    datasets = [d for d in datasets if d in curve_df["dataset"].unique()]
    xticks = sorted(curve_df[lambda_col].dropna().unique().tolist())

    g = sns.relplot(
        kind="line",
        data=curve_df,
        x=lambda_col,
        y="noise_scale",
        hue="method",
        style="method",
        col="dataset",
        markers=True,
        dashes=False,
        hue_order=["GAP-EDPES"],
        style_order=["GAP-EDPES"],
        col_order=datasets,
        aspect=1.25,
        facet_kws={"sharey": False, "sharex": True},
    )

    g.set_axis_labels(r"Schedule parameter $(\lambda)$", "Noise scale")

    for ax, dataset in zip(g.axes[0], datasets):
        ax.set_title(dataset.capitalize(), fontdict={"fontweight": "bold"})
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", rotation=45)

    if g.legend is not None:
        g.legend.set_title("")

    g.fig.subplots_adjust(hspace=0.35, wspace=0.2)
    g.fig.set_figwidth(max(6 * len(datasets), 14))

    eps_tag = "all" if epsilon is None else str(epsilon).replace(".", "p")
    hops_tag = "all" if hops is None else str(hops).replace(".", "p")
    ref_tag = str(ref_lambda).replace(".", "p").replace("-", "m")

    pdf_path = os.path.join(out_dir, f"edge_gap_noise_scale_vs_lambda_strict_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.pdf")
    png_path = os.path.join(out_dir, f"edge_gap_noise_scale_vs_lambda_strict_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="./output/csv/GAP")
    parser.add_argument("--out_dir", type=str, default="./figs")
    parser.add_argument("--epsilon", type=float, default=None, help="只画某个 epsilon 的结果，例如 1")
    parser.add_argument("--hops", type=float, default=None, help="只画某个 hops 的结果，例如 2")
    parser.add_argument("--lambda_col", type=str, default="", help="lambda 列名；不传则自动猜")
    parser.add_argument("--datasets", type=str, default="", help='逗号分隔，例如 "facebook,reddit,amazon"')
    parser.add_argument("--ref_lambda", type=float, default=0.0, help="严格同参模式下，用哪个 lambda 作为参考配置，默认 0.0")
    args = parser.parse_args()

    df = load_local_csvs(args.csv_dir)

    print("\n[INFO] 读到的列：")
    print(sorted(df.columns.tolist()))

    print("\n[INFO] method 分布：")
    print(df["method"].value_counts(dropna=False))

    lambda_col = infer_lambda_col(df, args.lambda_col)
    print(f"\n[INFO] 使用 lambda 列: {lambda_col}")

    plot_accuracy_vs_lambda(
        df=df,
        out_dir=args.out_dir,
        lambda_col=lambda_col,
        epsilon=args.epsilon,
        hops=args.hops,
        datasets_raw=args.datasets,
        ref_lambda=args.ref_lambda,
    )

    try:
        plot_noise_scale_vs_lambda(
            df=df,
            out_dir=args.out_dir,
            lambda_col=lambda_col,
            epsilon=args.epsilon,
            hops=args.hops,
            datasets_raw=args.datasets,
            ref_lambda=args.ref_lambda,
        )
    except KeyError as e:
        print(f"[WARN] 跳过 noise_scale 图：{e}")


if __name__ == "__main__":
    main()