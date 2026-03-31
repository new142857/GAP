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
        "GAP-EDPE2": "GAP-EDPE2",
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
        "noise_scale": "noise_scale_mean",
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

    non_private_methods = {r"GAP-$\infty$", r"SAGE-$\infty$"}

    x = pd.to_numeric(df["epsilon"], errors="coerce")
    keep_eps = np.isclose(x, float(epsilon), atol=1e-12, rtol=0)
    keep_non_private = df["method"].isin(non_private_methods)

    return df[keep_eps | keep_non_private].copy()


def maybe_filter_hops(df: pd.DataFrame, hops):
    if hops is None:
        return df

    if "hops" not in df.columns:
        print("[WARN] 指定了 --hops，但数据里没有 hops 列，跳过该过滤。")
        return df

    x = pd.to_numeric(df["hops"], errors="coerce")
    keep_hops = np.isclose(x, float(hops), atol=1e-12, rtol=0)

    return df[keep_hops].copy()


def best_by_group(df: pd.DataFrame, group_cols):
    idx = df.groupby(group_cols)["val/acc_mean"].idxmax().values
    return df.loc[idx].reset_index(drop=True)


def infer_strict_match_cols(df: pd.DataFrame, lambda_col: str):
    exclude_cols = {
        "id", "method", "dataset",
        "project", "output_dir", "logger", "device", "data_dir",

        "train/acc_mean", "train/acc_std", "train/acc_ci",
        "val/acc_mean", "val/acc_std", "val/acc_ci",
        "test/acc_mean", "test/acc_std", "test/acc_ci",
        "train/loss_mean", "train/loss_std", "train/loss_ci",
        "val/loss_mean", "val/loss_std", "val/loss_ci",
        "duration_mean", "duration_std", "duration_ci",
        "epoch_mean", "epoch_std", "epoch_ci",

        "noise_scale_mean", "noise_scale_std", "noise_scale_ci",

        lambda_col,
        "hop_noise_lambda",
        "hop_noise_mode",
        "hop_noise_factors",
        "hop_noise_temp",
        "hop_noise_poly_power",
        "hop_noise_sigmoid_alpha",
        "hop_budget_weights",

        "seed",
    }

    return [c for c in df.columns if c not in exclude_cols]


def _series_match_value(series: pd.Series, value):
    if pd.isna(value):
        return series.isna()

    s_num = pd.to_numeric(series, errors="coerce")
    try:
        v_num = float(value)
        if s_num.notna().any():
            return np.isclose(s_num, v_num, atol=1e-12, rtol=0)
    except Exception:
        pass

    return series.astype(str).str.strip().str.lower() == str(value).strip().lower()


def filter_by_conditions(df: pd.DataFrame, cond: dict, match_cols):
    out = df.copy()
    for col in match_cols:
        if col not in out.columns or col not in cond:
            continue
        out = out[_series_match_value(out[col], cond[col])]
    return out.copy()


def pick_reference_row(
    df: pd.DataFrame,
    dataset: str,
    method: str,
    lambda_col: str,
    ref_lambda: float,
    curve_mode: str = "all",
):
    """
    在指定 dataset/method 下，找到 lambda=ref_lambda 的参考记录；
    若有多条，取 val/acc_mean 最大的那条。
    如果 method 是 curve_method 家族，则可进一步限制 hop_noise_mode。
    """
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)].copy()
    if sub.empty:
        return None

    curve_mode = normalize_curve_mode(curve_mode)
    if curve_mode not in {"all", "*"} and "hop_noise_mode" in sub.columns:
        sub = sub[
            sub["hop_noise_mode"].astype(str).str.strip().str.lower() == curve_mode
        ].copy()

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
    curve_mode: str = "all",
):
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
            curve_mode=curve_mode,
        )

        if ref_row is None:
            print(f"[WARN] dataset={dataset} 没找到 {curve_method} 在 lambda={ref_lambda} 的参考记录，跳过。")
            continue

        ref_rows[dataset] = ref_row
        cond = {c: ref_row[c] for c in match_cols if c in ref_row.index}

        curve_sub = df[(df["dataset"] == dataset) & (df["method"] == curve_method)].copy()

        curve_mode_norm = normalize_curve_mode(curve_mode)
        if curve_mode_norm not in {"all", "*"} and "hop_noise_mode" in curve_sub.columns:
            curve_sub = curve_sub[
                curve_sub["hop_noise_mode"].astype(str).str.strip().str.lower() == curve_mode_norm
            ].copy()

        curve_sub = filter_by_conditions(curve_sub, cond, match_cols)
        curve_sub[lambda_col] = pd.to_numeric(curve_sub[lambda_col], errors="coerce")
        curve_sub = curve_sub[curve_sub[lambda_col].notna()].copy()

        if not curve_sub.empty:
            curve_sub = best_by_group(curve_sub, ["dataset", "method", lambda_col])
            curve_parts.append(curve_sub)
        else:
            print(f"[WARN] dataset={dataset} 的 {curve_method} 严格同参曲线为空。")

        for base_method in baseline_methods:
            base_sub = df[(df["dataset"] == dataset) & (df["method"] == base_method)].copy()

            if base_method == "GAP-EDP":
                base_sub = filter_by_conditions(base_sub, cond, match_cols)

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

def normalize_curve_mode(curve_mode: str) -> str:
    if curve_mode is None:
        return "all"
    x = str(curve_mode).strip().lower()
    return x if x else "all"


def filter_curve_family_mode(df: pd.DataFrame, curve_method: str, curve_mode: str):
    """
    只对 curve_method 过滤 hop_noise_mode；
    baseline 方法（GAP-EDP / GAP-inf）不受影响。
    """
    curve_mode = normalize_curve_mode(curve_mode)

    if curve_mode in {"all", "*"}:
        return df.copy()

    if "hop_noise_mode" not in df.columns:
        print("[WARN] 数据里没有 hop_noise_mode 列，无法按 mode 过滤，跳过。")
        return df.copy()

    out = df.copy()
    curve_mask = out["method"] == curve_method
    mode_mask = (
        out["hop_noise_mode"]
        .astype(str)
        .str.strip()
        .str.lower()
        == curve_mode
    )

    return out[(~curve_mask) | (mode_mask)].copy()

def plot_accuracy_vs_lambda(df: pd.DataFrame, out_dir: str, lambda_col: str, epsilon=None, datasets_raw: str = "",
                            hops=None, ref_lambda: float = 0.0, show_gap_inf: bool = True,
                            curve_mode: str = "exp"):
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

    df = filter_curve_family_mode(df, curve_method="GAP-EDPES", curve_mode=curve_mode)

    datasets = parse_dataset_list(datasets_raw, df)

    keep_methods = ["GAP-EDPES", "GAP-EDP"]
    if show_gap_inf:
        keep_methods.append(r"GAP-$\infty$")
    df = df[df["method"].isin(keep_methods)].copy()
    df = df[df["dataset"].isin(datasets)].copy()

    baseline_methods = ["GAP-EDP"]
    if show_gap_inf:
        baseline_methods.append(r"GAP-$\infty$")

    curve_df, baseline_maps, ref_rows, match_cols = build_strict_family_data(
        df=df,
        datasets=datasets,
        curve_method="GAP-EDPES",
        baseline_methods=baseline_methods,
        lambda_col=lambda_col,
        ref_lambda=ref_lambda,
        curve_mode=curve_mode,
    )

    edp_base = baseline_maps.get("GAP-EDP", {})
    inf_base = baseline_maps.get(r"GAP-$\infty$", {}) if show_gap_inf else {}

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
        if show_gap_inf and dataset in inf_base:
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

    legend_candidates = ["GAP-EDPES", "GAP-EDP"]
    if show_gap_inf:
        legend_candidates.append(r"GAP-$\infty$")

    legend_order = [m for m in legend_candidates if m in legend_data]
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

    mode_tag = normalize_curve_mode(curve_mode).replace("*", "all")
    pdf_path = os.path.join(out_dir, f"edge_gap_acc_vs_lambda_strict_mode_{mode_tag}_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.pdf")
    png_path = os.path.join(out_dir, f"edge_gap_acc_vs_lambda_strict_mode_{mode_tag}_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.png")

    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def plot_noise_scale_vs_lambda(df: pd.DataFrame, out_dir: str, lambda_col: str, epsilon=None, datasets_raw: str = "",
                               hops=None, ref_lambda: float = 0.0, show_gap_inf: bool = True,
                               curve_mode: str = "exp"):
    if "noise_scale_mean" not in df.columns:
        raise KeyError("当前 CSV 没有 noise_scale_mean 列，请先把训练时的 self.noise_scale 写入结果 CSV。")

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

    df = filter_curve_family_mode(df, curve_method="GAP-EDPES", curve_mode=curve_mode)

    datasets = parse_dataset_list(datasets_raw, df)

    keep_methods = ["GAP-EDPES", "GAP-EDP"]
    if show_gap_inf:
        keep_methods.append(r"GAP-$\infty$")
    tmp_df = df[df["method"].isin(keep_methods)].copy()
    tmp_df = tmp_df[tmp_df["dataset"].isin(datasets)].copy()

    baseline_methods = ["GAP-EDP"]
    if show_gap_inf:
        baseline_methods.append(r"GAP-$\infty$")

    curve_df, _, _, _ = build_strict_family_data(
        df=tmp_df,
        datasets=datasets,
        curve_method="GAP-EDPES",
        baseline_methods=baseline_methods,
        lambda_col=lambda_col,
        ref_lambda=ref_lambda,
        curve_mode=curve_mode,
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

    mode_tag = normalize_curve_mode(curve_mode).replace("*", "all")
    pdf_path = os.path.join(out_dir, f"edge_gap_noise_scale_vs_lambda_strict_mode_{mode_tag}_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.pdf")
    png_path = os.path.join(out_dir, f"edge_gap_noise_scale_vs_lambda_strict_mode_{mode_tag}_eps_{eps_tag}_hops_{hops_tag}_ref_{ref_tag}.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def add_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "hop_noise_mode" not in df.columns:
        raise KeyError("缺少 hop_noise_mode 列，无法做 exp / softmax / poly / sigmoid 对比。")

    for c in [
        "hop_noise_lambda",
        "hop_noise_temp",
        "hop_noise_poly_power",
        "hop_noise_sigmoid_alpha",
    ]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["schedule_mode"] = df["hop_noise_mode"].astype(str).str.strip().str.lower()

    df["schedule_value"] = np.nan
    df.loc[df["schedule_mode"] == "exp", "schedule_value"] = df.loc[
        df["schedule_mode"] == "exp", "hop_noise_lambda"
    ]
    df.loc[df["schedule_mode"] == "softmax", "schedule_value"] = df.loc[
        df["schedule_mode"] == "softmax", "hop_noise_temp"
    ]
    df.loc[df["schedule_mode"] == "poly", "schedule_value"] = df.loc[
        df["schedule_mode"] == "poly", "hop_noise_poly_power"
    ]
    df.loc[df["schedule_mode"] == "sigmoid", "schedule_value"] = df.loc[
        df["schedule_mode"] == "sigmoid", "hop_noise_sigmoid_alpha"
    ]

    df["schedule_param_name"] = ""
    df.loc[df["schedule_mode"] == "exp", "schedule_param_name"] = r"$\lambda$"
    df.loc[df["schedule_mode"] == "softmax", "schedule_param_name"] = r"$T$"
    df.loc[df["schedule_mode"] == "poly", "schedule_param_name"] = r"$p$"
    df.loc[df["schedule_mode"] == "sigmoid", "schedule_param_name"] = r"$\alpha$"

    df["schedule_label"] = df["schedule_mode"]
    df.loc[df["schedule_mode"] == "exp", "schedule_label"] = (
        "exp(" + r"$\lambda$" + "=" + df.loc[df["schedule_mode"] == "exp", "schedule_value"].astype(str) + ")"
    )
    df.loc[df["schedule_mode"] == "softmax", "schedule_label"] = (
        "softmax(" + r"$T$" + "=" + df.loc[df["schedule_mode"] == "softmax", "schedule_value"].astype(str) + ")"
    )
    df.loc[df["schedule_mode"] == "poly", "schedule_label"] = (
        "poly(" + r"$p$" + "=" + df.loc[df["schedule_mode"] == "poly", "schedule_value"].astype(str) + ")"
    )
    df.loc[df["schedule_mode"] == "sigmoid", "schedule_label"] = (
        "sigmoid(" + r"$\alpha$" + "=" + df.loc[df["schedule_mode"] == "sigmoid", "schedule_value"].astype(str) + ")"
    )

    return df


def maybe_filter_modes(df: pd.DataFrame, modes_raw: str):
    if not modes_raw.strip():
        return df
    modes = [x.strip().lower() for x in modes_raw.split(",") if x.strip()]
    return df[df["schedule_mode"].isin(modes)].copy()


def get_best_baseline_by_dataset(df: pd.DataFrame, method_name: str):
    sub = df[df["method"] == method_name].copy()
    if sub.empty:
        return {}
    sub = best_by_group(sub, ["dataset", "method"])
    return sub.set_index("dataset")["test/acc_mean"].to_dict()


def get_best_schedule_points(df: pd.DataFrame):
    if df.empty:
        return df.copy()

    point_df = best_by_group(df, ["dataset", "schedule_mode", "schedule_value"])
    idx = point_df.groupby(["dataset", "schedule_mode"])["val/acc_mean"].idxmax().values
    return point_df.loc[idx].reset_index(drop=True)


def plot_accuracy_vs_schedule(
    df: pd.DataFrame,
    out_dir: str,
    method_name: str = "GAP-EDPES",
    epsilon=None,
    datasets_raw: str = "",
    hops=None,
    modes_raw: str = "exp,softmax,poly,sigmoid",
    show_gap_inf: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.6,
        rc={
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "lines.linewidth": 2.5,
            "lines.markersize": 7,
        },
    )

    df = df.copy()
    if epsilon is not None:
        df = maybe_filter_epsilon(df, epsilon)
    if hops is not None:
        df = maybe_filter_hops(df, hops)

    df = add_schedule_columns(df)
    df = maybe_filter_modes(df, modes_raw)

    datasets = parse_dataset_list(datasets_raw, df)

    keep_methods = [method_name, "GAP-EDP"]
    if show_gap_inf:
        keep_methods.append(r"GAP-$\infty$")

    df = df[df["method"].isin(keep_methods)].copy()
    df = df[df["dataset"].isin(datasets)].copy()

    curve_df = df[df["method"] == method_name].copy()
    curve_df = curve_df[curve_df["schedule_value"].notna()].copy()

    if curve_df.empty:
        raise ValueError(f"没有找到 {method_name} 的 schedule 数据。")

    curve_df = best_by_group(curve_df, ["dataset", "schedule_mode", "schedule_value"])

    edp_base = get_best_baseline_by_dataset(df, "GAP-EDP")
    inf_base = get_best_baseline_by_dataset(df, r"GAP-$\infty$") if show_gap_inf else {}

    mode_order = [x.strip().lower() for x in modes_raw.split(",") if x.strip()]
    dataset_order = [d for d in datasets if d in curve_df["dataset"].unique()]

    row_order = [m for m in mode_order if m in curve_df["schedule_mode"].unique()]

    g = sns.FacetGrid(
        curve_df,
        row="schedule_mode",
        col="dataset",
        row_order=row_order,
        col_order=dataset_order,
        sharex=False,
        sharey=False,
        height=4.0,
        aspect=1.2,
        margin_titles=True,
    )
    g.map_dataframe(sns.lineplot, x="schedule_value", y="test/acc_mean", marker="o")

    for i, mode in enumerate(g.row_names):
        for j, dataset in enumerate(g.col_names):
            ax = g.axes[i, j]

            if dataset in edp_base:
                ax.axhline(
                    edp_base[dataset],
                    linestyle="--",
                    color=sns.color_palette("deep")[2],
                    label="GAP-EDP",
                )

            if show_gap_inf and dataset in inf_base:
                ax.axhline(
                    inf_base[dataset],
                    linestyle=":",
                    color=sns.color_palette("deep")[3],
                    label=r"GAP-$\infty$",
                )

            sub = curve_df[
                (curve_df["schedule_mode"] == mode) &
                (curve_df["dataset"] == dataset)
            ].copy()

            if not sub.empty:
                xticks = sorted(sub["schedule_value"].dropna().unique().tolist())
                ax.set_xticks(xticks)
                ax.tick_params(axis="x", rotation=45)

            if mode == "exp":
                ax.set_xlabel(r"$\lambda$")
            elif mode == "softmax":
                ax.set_xlabel(r"$T$")
            elif mode == "poly":
                ax.set_xlabel(r"$p$")
            elif mode == "sigmoid":
                ax.set_xlabel(r"$\alpha$")
            else:
                ax.set_xlabel("schedule value")

            if j == 0:
                ax.set_ylabel("Accuracy (%)")
            else:
                ax.set_ylabel("")

    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    handles = []
    labels = []
    if dataset_order:
        handles, labels = g.axes[0, 0].get_legend_handles_labels()

    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        g.fig.legend(
            handles=list(uniq.values()),
            labels=list(uniq.keys()),
            loc="upper center",
            ncol=len(uniq),
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )

    g.fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.22)

    eps_tag = "all" if epsilon is None else str(epsilon).replace(".", "p")
    hops_tag = "all" if hops is None else str(hops).replace(".", "p")

    safe_method_name = method_name.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    pdf_path = os.path.join(out_dir, f"schedule_scan_{safe_method_name}_eps_{eps_tag}_hops_{hops_tag}.pdf")
    png_path = os.path.join(out_dir, f"schedule_scan_{safe_method_name}_eps_{eps_tag}_hops_{hops_tag}.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")


def plot_best_accuracy_by_mode(
    df: pd.DataFrame,
    out_dir: str,
    method_name: str = "GAP-EDPES",
    epsilon=None,
    datasets_raw: str = "",
    hops=None,
    modes_raw: str = "exp,softmax,poly,sigmoid",
    show_gap_inf: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    sns.set(
        context="paper",
        style="ticks",
        palette="deep",
        font_scale=1.6,
        rc={
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
        },
    )

    df = df.copy()
    if epsilon is not None:
        df = maybe_filter_epsilon(df, epsilon)
    if hops is not None:
        df = maybe_filter_hops(df, hops)

    df = add_schedule_columns(df)
    df = maybe_filter_modes(df, modes_raw)

    datasets = parse_dataset_list(datasets_raw, df)

    keep_methods = [method_name, "GAP-EDP"]
    if show_gap_inf:
        keep_methods.append(r"GAP-$\infty$")

    df = df[df["method"].isin(keep_methods)].copy()
    df = df[df["dataset"].isin(datasets)].copy()

    curve_df = df[df["method"] == method_name].copy()
    curve_df = curve_df[curve_df["schedule_value"].notna()].copy()

    if curve_df.empty:
        raise ValueError(f"没有找到 {method_name} 的 schedule 数据。")

    best_df = get_best_schedule_points(curve_df)

    edp_base = get_best_baseline_by_dataset(df, "GAP-EDP")
    inf_base = get_best_baseline_by_dataset(df, r"GAP-$\infty$") if show_gap_inf else {}

    mode_order = [x.strip().lower() for x in modes_raw.split(",") if x.strip()]
    dataset_order = [d for d in datasets if d in best_df["dataset"].unique()]

    g = sns.catplot(
        data=best_df,
        kind="bar",
        x="schedule_mode",
        y="test/acc_mean",
        col="dataset",
        col_order=dataset_order,
        order=[m for m in mode_order if m in best_df["schedule_mode"].unique()],
        sharey=False,
        height=4.2,
        aspect=1.2,
    )

    for ax, dataset in zip(g.axes[0], dataset_order):
        if dataset in edp_base:
            ax.axhline(
                edp_base[dataset],
                linestyle="--",
                color=sns.color_palette("deep")[2],
                label="GAP-EDP",
            )

        if show_gap_inf and dataset in inf_base:
            ax.axhline(
                inf_base[dataset],
                linestyle=":",
                color=sns.color_palette("deep")[3],
                label=r"GAP-$\infty$",
            )

        ax.set_xlabel("schedule mode")
        ax.set_ylabel("Best Accuracy (%)")
        ax.tick_params(axis="x", rotation=20)

    handles, labels = g.axes[0][0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        g.fig.legend(
            handles=list(uniq.values()),
            labels=list(uniq.keys()),
            loc="upper center",
            ncol=len(uniq),
            bbox_to_anchor=(0.5, 1.02),
            frameon=True,
        )

    g.fig.subplots_adjust(top=0.86, wspace=0.20)

    eps_tag = "all" if epsilon is None else str(epsilon).replace(".", "p")
    hops_tag = "all" if hops is None else str(hops).replace(".", "p")

    safe_method_name = method_name.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
    pdf_path = os.path.join(out_dir, f"schedule_best_compare_{safe_method_name}_eps_{eps_tag}_hops_{hops_tag}.pdf")
    png_path = os.path.join(out_dir, f"schedule_best_compare_{safe_method_name}_eps_{eps_tag}_hops_{hops_tag}.png")
    g.savefig(pdf_path, bbox_inches="tight")
    g.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close("all")

    print(f"[OK] 已保存: {pdf_path}")
    print(f"[OK] 已保存: {png_path}")

    print("\n[INFO] 各 dataset / mode 的最优点：")
    show_cols = [
        "dataset", "schedule_mode", "schedule_value",
        "val/acc_mean", "test/acc_mean",
        "hop_noise_lambda", "hop_noise_temp",
        "hop_noise_poly_power", "hop_noise_sigmoid_alpha"
    ]
    show_cols = [c for c in show_cols if c in best_df.columns]
    print(best_df[show_cols].sort_values(["dataset", "schedule_mode"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, default="./output/csv/GAP")
    parser.add_argument("--out_dir", type=str, default="./figs")
    parser.add_argument("--epsilon", type=float, default=None, help="只画某个 epsilon 的结果，例如 1")
    parser.add_argument("--hops", type=float, default=None, help="只画某个 hops 的结果，例如 2")
    parser.add_argument("--lambda_col", type=str, default="", help="lambda 列名；不传则自动猜")
    parser.add_argument("--datasets", type=str, default="", help='逗号分隔，例如 "facebook,reddit,amazon"')
    parser.add_argument("--ref_lambda", type=float, default=0.0, help="严格同参模式下，用哪个 lambda 作为参考配置，默认 0.0")
    parser.add_argument("--hide_gap_inf", action="store_true", help="传入后不绘制 GAP-inf 基线")

    parser.add_argument("--method_name", type=str, default="GAP-EDPES", help="schedule 对比时要比较的主方法名")
    parser.add_argument("--modes", type=str, default="exp,softmax,poly,sigmoid", help='要比较的 mode，逗号分隔')
    parser.add_argument("--plot_schedule_compare", action="store_true", help="额外画 exp/softmax/poly/sigmoid 对比图")
    parser.add_argument("--lambda_mode", type=str, default="exp",help='原始 lambda 图只使用哪种 hop_noise_mode，默认 exp；如需不过滤可传 "all"')

    args = parser.parse_args()

    df = load_local_csvs(args.csv_dir)

    print("\n[INFO] 读到的列：")
    print(sorted(df.columns.tolist()))

    print("\n[INFO] method 分布：")
    print(df["method"].value_counts(dropna=False))

    try:
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
            show_gap_inf=not args.hide_gap_inf,
            curve_mode=args.lambda_mode,
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
                show_gap_inf=not args.hide_gap_inf,
                curve_mode=args.lambda_mode,
            )
        except KeyError as e:
            print(f"[WARN] 跳过 noise_scale 图：{e}")

    except Exception as e:
        print(f"[WARN] 跳过原始 lambda 图：{e}")

    if args.plot_schedule_compare:
        plot_accuracy_vs_schedule(
            df=df,
            out_dir=args.out_dir,
            method_name=normalize_method(args.method_name),
            epsilon=args.epsilon,
            hops=args.hops,
            datasets_raw=args.datasets,
            modes_raw=args.modes,
            show_gap_inf=not args.hide_gap_inf,
        )

        plot_best_accuracy_by_mode(
            df=df,
            out_dir=args.out_dir,
            method_name=normalize_method(args.method_name),
            epsilon=args.epsilon,
            hops=args.hops,
            datasets_raw=args.datasets,
            modes_raw=args.modes,
            show_gap_inf=not args.hide_gap_inf,
        )


if __name__ == "__main__":
    main()