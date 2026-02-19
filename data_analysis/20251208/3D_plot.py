#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import zipfile
import glob

import numpy as np
import plotly.express as px


def extract_zip(zip_path, out_dir=None):
    """解压 zip，返回解压目录路径。"""
    if out_dir is None:
        base = os.path.splitext(os.path.basename(zip_path))[0]
        out_dir = os.path.join(os.path.dirname(zip_path), base)

    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    return out_dir


def find_base_prefix(extract_dir):
    """
    在解压目录里找 *_e_trace 和 *_trapped_signal，
    返回公共前缀 base，例如 20251208_173801。
    """
    e_trace_files = glob.glob(os.path.join(extract_dir, "*_e_trace"))
    if not e_trace_files:
        raise FileNotFoundError("找不到 *_e_trace 文件")

    # 先用第一个 e_trace 的名字确定前缀
    e_path = e_trace_files[0]
    base_prefix = e_path.rsplit("_e_trace", 1)[0]

    # 检查对应的 trapped_signal
    trapped_path = base_prefix + "_trapped_signal"
    if not os.path.exists(trapped_path):
        raise FileNotFoundError(f"找不到对应的 {trapped_path}")

    return base_prefix, e_path, trapped_path


def load_data(base_prefix, e_path, trapped_path):
    """读取 e_trace (Ex,Ey,Ez) 和 trapped_signal。"""
    e_trace = np.loadtxt(e_path, delimiter=",")
    trapped = np.loadtxt(trapped_path)

    if e_trace.ndim != 2 or e_trace.shape[1] < 3:
        raise ValueError("e_trace 格式不对，至少要有三列 Ex,Ey,Ez")

    if trapped.shape[0] != e_trace.shape[0]:
        raise ValueError(
            f"trapped_signal 行数({trapped.shape[0]}) "
            f"和 e_trace 行数({e_trace.shape[0]}) 不一致"
        )

    Ex = e_trace[:, 0]
    Ey = e_trace[:, 1]
    Ez = e_trace[:, 2]

    return Ex, Ey, Ez, trapped


def make_interactive_3d(Ex, Ey, Ez, trapped, title=None):
    """用 plotly 画交互式 3D 散点图。"""
    import pandas as pd

    df = pd.DataFrame(
        {"Ex": Ex, "Ey": Ey, "Ez": Ez, "trapped_signal": trapped}
    )

    fig = px.scatter_3d(
        df,
        x="Ex",
        y="Ey",
        z="Ez",
        color="trapped_signal",
        color_continuous_scale="Viridis",
        hover_data=["trapped_signal"],
        title=title or "Ex–Ey–Ez 3D scatter (color = trapped_signal)",
    )
    fig.update_traces(marker=dict(size=4))

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="从 zip 文件里读 e_trace / trapped_signal 并画交互式 3D 图"
    )
    parser.add_argument(
        "zip_file",
        help="zip 文件名，例如 20251208_173801_.zip",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存 html，不在浏览器中打开",
    )
    parser.add_argument(
        "--html",
        default=None,
        help="输出 html 文件名（默认：<zip同名>.html）",
    )

    args = parser.parse_args()

    zip_path = os.path.abspath(args.zip_file)
    extract_dir = extract_zip(zip_path)

    base_prefix, e_path, trapped_path = find_base_prefix(extract_dir)
    Ex, Ey, Ez, trapped = load_data(base_prefix, e_path, trapped_path)

    fig = make_interactive_3d(
        Ex, Ey, Ez, trapped,
        title=os.path.basename(base_prefix),
    )

    # 保存为 HTML
    html_name = (
        args.html
        if args.html is not None
        else os.path.splitext(args.zip_file)[0] + "_3d.html"
    )
    html_path = os.path.abspath(html_name)
    fig.write_html(html_path)

    if not args.no_show:
        fig.show()


if __name__ == "__main__":
    main()
