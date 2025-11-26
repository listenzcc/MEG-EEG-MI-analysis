"""
File: summary.source.tfr.py
Author: Chuncheng Zhang
Date: 2025-11-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary the source tfr graphs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-26 ------------------------
# Requirements and constants
from itertools import product
import matplotlib.gridspec as gridspec
from util.easy_import import *

# %%
TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}

DATA_DIR = Path('./data/TFR-Source')

# %% ---- 2025-11-26 ------------------------
# Function and class


# %% ---- 2025-11-26 ------------------------
# Play ground
# 定义分类
subjects = ['S07', 'average']
modalities = ['meg']
bands = ['beta', 'alpha']
tasks = ['1', '2', '3', '4', '5']
times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]  # 根据文件名中的时间

# img.shape is (800, 1600, 3)

for subject, mode, band in tqdm(product(subjects, modalities, bands), total=4):
    print(subject, mode, band)
    fig, axes = plt.subplots(len(tasks), len(times), figsize=(16, 10))
    for i, t in enumerate(times):
        for j, task in enumerate(tasks):
            ax = axes[j][i]
            title = f'{t=:0.1f}'
            fname = f'{subject}-{mode}-{task}-{band}-{t:0.1f}.png'
            img = plt.imread(DATA_DIR / fname)
            img = img[100:700, :800]
            ax.imshow(img)
            if j == 0:
                ax.set_title(title, fontweight='bold')
            ax.axis('off')
            # 在第一列添加任务名称
            if i == 0:
                ax.text(-0.2, 0.5, TASK_TABLE[task],
                        rotation=90,
                        transform=ax.transAxes,
                        fontsize=12, fontweight='bold',
                        va='center', ha='right')
    fig.suptitle(
        f'TFR in source | {subject.title()} | {band.title()}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(DATA_DIR / f'results-{subject}-{mode}-{band}.png')
    plt.show()


# %% ---- 2025-11-26 ------------------------
# Pending


# %% ---- 2025-11-26 ------------------------
# Pending
