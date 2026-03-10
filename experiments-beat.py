from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

# 你原来用的 console 也可以继续用；如果不想依赖 core，就改成 print
from core import console


def _as_list(v: Any) -> list[Any]:
    return v if isinstance(v, list) else [v]


def _expand_grid(kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """
    把 kwargs 中 value 为 list 的参数做笛卡尔积展开，生成多个 job 参数组合。
    例：{"epsilon":[1,2], "hops":[1,2]} -> 4 组组合
    """
    keys = list(kwargs.keys())
    values_lists = [_as_list(kwargs[k]) for k in keys]
    combos = []
    for vals in product(*values_lists):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


@dataclass
class LocalJobRegistry:
    """
    替代 WandBJobRegistry：只负责把 register(...) 展开为命令字符串列表
    """
    python_exe: str = "python"
    job_list: list[str] = field(default_factory=list)

    def register(self, script: str, method: str, *positional: str, **kwargs: Any) -> None:
        # 对 kwargs 做 grid 展开
        for combo in _expand_grid(kwargs):
            cmd = [self.python_exe, script, method, *positional]
            for k, v in combo.items():
                if v is None:
                    continue
                # 统一用 --key value 的形式（和你现在 train.py 的用法一致）
                cmd.append(f"--{k}")
                cmd.append(str(v))
            self.job_list.append(" ".join(shlex.quote(x) for x in cmd))


def create_train_commands(registry: LocalJobRegistry) -> list[str]:
    # ### Hyper-parameters
    datasets = ['facebook', 'reddit', 'amazon']
    batch_size = {'facebook': 256, 'reddit': 2048, 'amazon': 4096}

    gap_methods  = ['gap-inf', 'gap-edp', 'gap-ndp']
    sage_methods = ['sage-inf', 'sage-edp', 'sage-ndp']
    mlp_methods  = ['mlp', 'mlp-dp']
    inf_methods  = ['gap-inf', 'sage-inf']
    edp_methods  = ['gap-edp', 'sage-edp', 'mlp']
    ndp_methods  = ['gap-ndp', 'sage-ndp', 'mlp-dp']
    all_methods  = inf_methods + edp_methods + ndp_methods
    hparams = {dataset: {method: {} for method in all_methods} for dataset in datasets}

    for dataset in datasets:
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['encoder_layers'] = 2
            hparams[dataset][method]['base_layers'] = 1
            hparams[dataset][method]['head_layers'] = 1
            hparams[dataset][method]['combine'] = 'cat'
            hparams[dataset][method]['hops'] = [1, 2, 3, 4, 5]
        # For SAGE methods
        for method in sage_methods:
            hparams[dataset][method]['base_layers'] = 2
            hparams[dataset][method]['head_layers'] = 1
            if method != 'sage-ndp':
                hparams[dataset][method]['mp_layers'] = [1, 2, 3, 4, 5]
        # For MLP methods
        for method in mlp_methods:
            hparams[dataset][method]['num_layers'] = 3
        # For GAP-NDP and SAGE-NDP
        for method in ['gap-ndp', 'sage-ndp']:
            hparams[dataset][method]['max_degree'] = [100, 200, 300, 400]
        # For all methods
        for method in all_methods:
            hparams[dataset][method]['hidden_dim'] = 16
            hparams[dataset][method]['activation'] = 'selu'
            hparams[dataset][method]['optimizer'] = 'adam'
            hparams[dataset][method]['learning_rate'] = 0.01
            hparams[dataset][method]['repeats'] = 10
            if method in ndp_methods:
                hparams[dataset][method]['max_grad_norm'] = 1
                hparams[dataset][method]['epochs'] = 10
                hparams[dataset][method]['batch_size'] = batch_size[dataset]
            else:
                hparams[dataset][method]['batch_norm'] = True
                hparams[dataset][method]['epochs'] = 100
                hparams[dataset][method]['batch_size'] = 'full'
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['encoder_epochs'] = hparams[dataset][method]['epochs']

    # ### Accuracy/Privacy Trade-off
    for dataset in datasets:
        for method in all_methods:
            params = {}
            if method in ndp_methods:
                params['epsilon'] = [1, 2, 4, 8, 16]
            elif method in ['gap-edp', 'sage-edp']:
                params['epsilon'] = [0.1, 0.2, 0.5, 1, 2, 4, 8]

            registry.register(
                'train.py',
                method,
                dataset=dataset,
                **params,
                **hparams[dataset][method]
            )

    # ### Effect of Encoder
    for dataset in datasets:
        for method in ['gap-edp', 'gap-ndp']:
            hp = {**hparams[dataset][method]}
            default_encoder_layers = hp.pop('encoder_layers')
            epsilon = [0.5, 1, 2, 4, 8] if method == 'gap-edp' else [1, 2, 4, 8, 16]
            registry.register(
                'train.py',
                method,
                dataset=dataset,
                encoder_layers=[0, default_encoder_layers],
                epsilon=epsilon,
                **hp
            )

    # ### Effect of Hops
    for dataset in datasets:
        for method in ['gap-edp', 'gap-ndp']:
            hp = {**hparams[dataset][method]}
            hp.pop('hops')
            hops = [1, 2, 3, 4, 5]
            epsilon = [1, 2, 4, 8] if method == 'gap-edp' else [2, 4, 8, 16]
            registry.register(
                'train.py',
                method,
                dataset=dataset,
                hops=hops,
                epsilon=epsilon,
                **hp
            )

    # ### Effect of Degree
    for dataset in datasets:
        method = 'gap-ndp'
        hp = {**hparams[dataset][method]}
        hp.pop('max_degree')
        max_degree = [10, 20, 50, 100, 200, 300, 400]
        epsilon = [2, 4, 8, 16]
        registry.register(
            'train.py',
            method,
            dataset=dataset,
            max_degree=max_degree,
            epsilon=epsilon,
            **hp
        )

    return registry.job_list


def create_attack_commands(registry: LocalJobRegistry) -> list[str]:
    # Hyperparameters
    datasets = ['facebook', 'reddit', 'amazon']
    gap_methods  = ['gap-inf', 'gap-ndp']
    sage_methods = ['sage-inf', 'sage-ndp']
    mlp_methods  = ['mlp', 'mlp-dp']
    ndp_methods  = ['gap-ndp', 'sage-ndp', 'mlp-dp']
    all_methods  = gap_methods + sage_methods + mlp_methods
    hparams = {dataset: {method: {} for method in all_methods} for dataset in datasets}

    for dataset in datasets:
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['shadow_encoder_layers'] = 2
            hparams[dataset][method]['shadow_base_layers'] = 1
            hparams[dataset][method]['shadow_head_layers'] = 1
            hparams[dataset][method]['shadow_combine'] = 'cat'
            hparams[dataset][method]['shadow_hops'] = 2
        # For SAGE methods
        for method in sage_methods:
            hparams[dataset][method]['shadow_base_layers'] = 2
            hparams[dataset][method]['shadow_head_layers'] = 1
            if method != 'sage-ndp':
                hparams[dataset][method]['shadow_mp_layers'] = 2
        # For MLP methods
        for method in mlp_methods:
            hparams[dataset][method]['shadow_num_layers'] = 3
        # For GAP-NDP and SAGE-NDP
        for method in ['gap-ndp', 'sage-ndp']:
            hparams[dataset][method]['shadow_max_degree'] = 100
        # For all methods
        for method in all_methods:
            hparams[dataset][method]['shadow_hidden_dim'] = 64
            hparams[dataset][method]['shadow_activation'] = 'selu'
            hparams[dataset][method]['shadow_optimizer'] = 'adam'
            hparams[dataset][method]['shadow_learning_rate'] = 0.01
            if method in ndp_methods:
                hparams[dataset][method]['shadow_max_grad_norm'] = 1
                hparams[dataset][method]['shadow_epochs'] = 10
                hparams[dataset][method]['shadow_batch_size'] = 256
            else:
                hparams[dataset][method]['shadow_batch_norm'] = True
                hparams[dataset][method]['shadow_epochs'] = 100
                hparams[dataset][method]['shadow_batch_size'] = 'full'
            if method != 'sage-ndp':
                hparams[dataset][method]['shadow_val_interval'] = 0
            if method in gap_methods:
                hparams[dataset][method]['shadow_encoder_epochs'] = hparams[dataset][method]['shadow_epochs']
            hparams[dataset][method]['num_nodes_per_class'] = 1000
            hparams[dataset][method]['attack_hidden_dim'] = 64
            hparams[dataset][method]['attack_num_layers'] = 3
            hparams[dataset][method]['attack_activation'] = 'selu'
            hparams[dataset][method]['attack_batch_norm'] = True
            hparams[dataset][method]['attack_batch_size'] = 'full'
            hparams[dataset][method]['attack_epochs'] = 100
            hparams[dataset][method]['attack_optimizer'] = 'adam'
            hparams[dataset][method]['attack_learning_rate'] = 0.01
            hparams[dataset][method]['attack_val_interval'] = 1
            hparams[dataset][method]['repeats'] = 10

    for dataset in datasets:
        for method in all_methods:
            params = {}
            if method in ndp_methods:
                params['shadow_epsilon'] = [1, 2, 4, 8, 16]

            # attack.py 的 positional 参数你原来是 ('nmi')，保持一致
            registry.register(
                'attack.py',
                method,
                'nmi',
                dataset=dataset,
                **params,
                **hparams[dataset][method]
            )

    return registry.job_list


def generate(path: str, python_exe: str = "python") -> None:
    """
    本地生成 jobs/experiments.sh（不依赖 wandb.yaml，不 pull）
    """
    reg_train = LocalJobRegistry(python_exe=python_exe)
    reg_attack = LocalJobRegistry(python_exe=python_exe)

    with console.status("generating job commands (local grid)..."):
        train_commands = create_train_commands(reg_train)
        attack_commands = create_attack_commands(reg_attack)

    job_list = train_commands + attack_commands
    console.info(f"{len(job_list)} jobs generated")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with console.status(f"saving jobs to {path}"):
        with open(p, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\n\n\n")
            for cmd in job_list:
                f.write(cmd + "\n")


def run_local(job_file: str) -> None:
    """
    本地顺序执行 job_file 里的命令（忽略空行/注释）
    """
    lines = Path(job_file).read_text(encoding="utf-8").splitlines()
    cmds = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        cmds.append(s)

    console.info(f"[LOCAL] Found {len(cmds)} commands in {job_file}")
    for i, cmd in enumerate(cmds, 1):
        console.rule(f"[LOCAL] ({i}/{len(cmds)}) {cmd}")
        subprocess.run(shlex.split(cmd), check=True)


def main() -> None:
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--generate", action="store_true", help="Generate jobs (local)")
    parser.add_argument("--run", action="store_true", help="Run jobs locally (sequential)")
    parser.add_argument("--path", type=str, default="jobs/experiments.sh", help="Path to the job file")
    parser.add_argument("--python", type=str, default="python", help="Python executable used in generated commands")
    args = parser.parse_args()

    if args.generate:
        generate(args.path, python_exe=args.python)
    if args.run:
        run_local(args.path)

    if not args.generate and not args.run:
        parser.error("Please specify either --generate or --run")


if __name__ == "__main__":
    main()
