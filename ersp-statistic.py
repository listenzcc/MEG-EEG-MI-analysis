"""
File: ersp-statistic.py
Author: Chuncheng Zhang
Date: 2025-06-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Statistical analysis for the ERSP.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-06-05 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as smt  # 用于多重比较校正

from rich import print
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import ttest_1samp

# %% ---- 2025-06-05 ------------------------
# Function and class


def find_files(root: Path, pattern: 'str') -> list:
    files = list(root.rglob(pattern))
    print(f'Found {len(files)} files as pattern: {pattern}')
    print(files)
    return files


# %% ---- 2025-06-05 ------------------------
# Play ground
found = find_files(Path('./data/h5'), 'ERSP_eeg-df.h5')
ERSP_eeg_dfs = [pd.read_hdf(f)
                for f in tqdm(found, 'Fetch EEG Data')]
print(ERSP_eeg_dfs[0])

found = find_files(Path('./data/h5'), 'ERSP_meg-df.h5')
ERSP_meg_dfs = [pd.read_hdf(f)
                for f in tqdm(found, 'Fetch MEG Data')]
print(ERSP_meg_dfs[0])


# %%
# 确保 time 和 freq 是数值型
output_fname = 'ERSP-EEG.png'
df = pd.concat(ERSP_eeg_dfs, axis=0).copy()

df["time"] = pd.to_numeric(df["time"])
df["freq"] = pd.to_numeric(df["freq"])

# 假设 df 是你的 DataFrame
# 确保 time 和 freq 是数值型
df["time"] = pd.to_numeric(df["time"])
df["freq"] = pd.to_numeric(df["freq"])
conditions = list(df['condition'].unique())
channels = list(df['channel'].unique())

freq_bounds = {"Delta": 3, "Theta": 7, "Alpha": 13, "Beta": 35, "Gamma": 140}

fig, axes = plt.subplots(len(conditions), len(channels), figsize=(12, 12))

# 按 condition 和 channel 分组，每组生成一个 heatmap
for i_cond, condition in enumerate(conditions):
    for i_chan, channel in enumerate(channels):
        # 筛选当前 condition 和 channel 的数据
        subset = df[(df["condition"] == condition)
                    & (df["channel"] == channel)]

        subset = subset.groupby(['condition', 'freq', 'time', 'channel', 'ch_type'], observed=True)[
            'value'].mean().reset_index()

        # 透视数据：行=freq，列=time，值=value
        pivot_table = subset.pivot(
            index="freq", columns="time", values="value")

        # 绘制 heatmap
        ax = axes[i_cond, i_chan]
        sns.heatmap(
            pivot_table,
            cmap="RdBu_r",  # 红蓝渐变色
            vmin=-1,        # 颜色范围最小值
            vmax=1,        # 颜色范围最大值
            cbar_kws={"label": "Value"},
            ax=ax
        )
        ax.set_xticks([pivot_table.columns.get_loc(e)
                      for e in [0, 1, 2, 3, 4]], labels=[0, 1, 2, 3, 4])

        ax.invert_yaxis()

        # 添加标题和标签
        ax.set_title(f"Condition {condition}, Channel {channel}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

plt.tight_layout()
plt.savefig(output_fname)
plt.show()

# %%

# %%

# %%


# %%
# %% ---- 2025-06-05 ------------------------
# Pending


# %% ---- 2025-06-05 ------------------------
# Pending

# %%
