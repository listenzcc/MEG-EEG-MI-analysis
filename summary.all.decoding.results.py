"""
File: summary.all.decoding.results.py
Author: Chuncheng Zhang
Date: 2025-11-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Summary all the decoding results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-26 ------------------------
# Requirements and constants
import pingouin as pg
from util.easy_import import *

OUTPUT_DIR = Path('./data')

# %% ---- 2025-11-26 ------------------------
# Function and class


def read_csv(fname):
    df = pd.read_csv(fname, index_col=0)
    return fix_column_contents(df)


def fix_column_contents(df):
    df['mode'] = df['mode'].map(lambda e: e.upper())
    # df['mode'] = df['mode'].map(lambda e: 'COMBINE' if e == 'ALL' else e)
    df['subject'] = df['subject'].map(lambda e: e[:3].upper())

    e = 'confusionMatrix'
    if e in df.columns:
        df[e] = df[e].map(string_to_numpy)

    return df


def string_to_numpy(array_str):
    import re
    # 匹配所有行
    rows = re.findall(r'\[([^\]]+)\]', array_str)
    array_data = []

    for row in rows:
        # 提取每行中的数字
        numbers = re.findall(r'[\d.]+', row)
        if numbers:
            array_data.append([float(num) for num in numbers])

    return np.array(array_data)


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.3, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=9, va='bottom')
    return

# %% ---- 2025-11-26 ------------------------
# Play ground


# %%
acc_fbcsp = read_csv('./data/decoding-fbcsp.csv')
acc_fbcnet = read_csv('../MEG-EEG-MI-fbcnet/data/decoding-fbcnet.csv')
acc_fbcnet_time = read_csv(
    '../MEG-EEG-MI-fbcnet/data/decoding-fbcnet-by-times.csv')
display(acc_fbcsp)
display(acc_fbcnet)
display(acc_fbcnet_time)

# %%
acc_freq = read_csv('./data/decoding-on-freq.csv')
acc_time = read_csv('./data/decoding-on-time.csv')
display(acc_freq)
display(acc_time)

# %%
# Combine acc_fbcnet_time and acc_time
acc_fbcnet_time['t'] = acc_fbcnet_time['tmax']
acc_time = pd.concat([acc_time, acc_fbcnet_time])
acc_time = acc_time[acc_time['method'].map(lambda e: e != 'accumulating')]
display(acc_time)

# %%
conf_fbcsp = read_csv('./data/decoding-fbcsp-confusion-matrix.csv')
conf_fbcnet = read_csv(
    '../MEG-EEG-MI-fbcnet/data/decoding-fbcnet-confusion-matrix.csv')
display(conf_fbcsp)
display(conf_fbcnet)


# %%
sns.set_theme(context='paper', style='ticks', font_scale=1)

TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}

# %%
df = pd.concat([conf_fbcsp, conf_fbcnet])

fig, axes = plt.subplots(2, 4, figsize=(15, 6), dpi=200)
ticklabels = list(TASK_TABLE.values())
# Absolute
for i, mode in enumerate(['EEG', 'MEG', 'COMBINE']):
    for j, method in enumerate(['FBCSP', 'FBCNet']):
        mat = np.mean(df.query(f'mode=="{mode}" & method=="{method}"')[
                      'confusionMatrix'].to_numpy(), axis=0)
        ax = axes[j][i]
        g = sns.heatmap(mat, vmin=0.1, vmax=0.6, annot=True,
                        cmap='Blues',
                        xticklabels=ticklabels if j == 1 else False,
                        yticklabels=ticklabels if i == 0 else False,
                        fmt='.2f',
                        annot_kws={'color': 'white'},
                        cbar=True,
                        cbar_kws={"shrink": 0.7},
                        ax=ax)
        g.set_xticklabels(g.get_xticklabels(), rotation=0)
        ax.set_title(f'{mode=} | {method=}', fontweight='bold')
        add_top_left_notion(ax, 'abcdefg'[j*3+i])

# Relative
i = 3
for j, method in enumerate(['FBCSP', 'FBCNet']):
    mat0 = np.mean(df.query(f'mode=="EEG" & method=="{method}"')[
        'confusionMatrix'].to_numpy(), axis=0)
    mat1 = np.mean(df.query(f'mode=="COMBINE" & method=="{method}"')[
        'confusionMatrix'].to_numpy(), axis=0)

    ax = axes[j, i]
    g = sns.heatmap(mat1-mat0,
                    vmin=-0.2, vmax=0.2, annot=True,
                    cmap='RdBu',
                    xticklabels=ticklabels if j == 1 else False,
                    yticklabels=ticklabels if i == 0 else False,
                    fmt='.2f',
                    annot_kws={'color': 'white'},
                    cbar=True,
                    cbar_kws={"shrink": 0.7},
                    ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=0)
    ax.set_title(f'COMBINE - EEG | {method=}', fontweight='bold')
    add_top_left_notion(ax, 'abcdefg'[j*3+i])


plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'confusion-matrix.png')
plt.show()

# %%

# Relative
# fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=200)
# ticklabels = list(TASK_TABLE.values())
# for i, mode in enumerate(['EEG', 'MEG', 'COMBINE']):
#     for j, method in enumerate(['FBCSP', 'FBCNet']):
#         mat = np.mean(df.query(f'mode=="{mode}" & method=="{method}"')[
#                       'confusionMatrix'].to_numpy(), axis=0)
#         mat -= base_mat
#         ax = axes[j][i]
#         g = sns.heatmap(mat, vmin=-0.2, vmax=0.2, annot=True,
#                         cmap='RdBu',
#                         xticklabels=ticklabels if j == 1 else False,
#                         yticklabels=ticklabels if i == 0 else False,
#                         fmt='.2f',
#                         annot_kws={'color': 'gray' if i ==
#                                    0 and j == 0 else 'white'},
#                         cbar=True,
#                         cbar_kws={"shrink": 0.7},
#                         ax=ax)
#         cbar = ax.collections[0].colorbar
#         cbar.set_ticks([-0.2, 0, 0.2])
#         g.set_xticklabels(g.get_xticklabels(), rotation=0)
#         ax.set_title(f'{mode=} | {method=}', fontweight='bold')
#         add_top_left_notion(ax, 'abcdefg'[j*3+i])
# plt.tight_layout()
# fig.savefig(OUTPUT_DIR / 'confusion-matrix-1.png')
# plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df = pd.concat([acc_fbcnet, acc_fbcsp])
ax = axes[0]
sns.barplot(df, x='method', hue='mode', y='accuracy',
            order=['FBCSP', 'FBCNet'],
            hue_order=['EEG', 'MEG', 'COMBINE'], ax=ax)
ax.axhline(y=0.2, linestyle=':', color='black')
# Add annotations for mean values
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 0.2,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='white')
ax.legend(loc='lower right', fontsize='small')
ax.set_title('Accuracy with FBCSP and FBCNet', fontweight='bold')

ax = axes[1]
sns.lineplot(acc_time, x='t', hue='mode', y='accuracy', style='method', ax=ax)
ax.axvline(x=0, linestyle=':', color='black')
ax.axhline(y=0.2, linestyle=':', color='black')
ax.set_ylim([0.15, 0.55])
ax.set_yticks([0.2, 0.3, 0.4, 0.5])
ax.legend(fontsize='small')
ax.set_title('Accuracy with LR across times', fontweight='bold')

for i, ax in enumerate(axes):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_top_left_notion(ax, notion='abcdefg'[i])

fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy.png')
plt.show()

# %% ---- 2025-11-26 ------------------------
# Pending
# 检查数据结构和因子水平
print("mode 水平:", df['mode'].unique())
print("method 水平:", df['method'].unique())
print("subject 数量:", df['subject'].nunique())


# 使用 pingouin 进行双因素重复测量方差分析
aov = pg.rm_anova(data=df, dv='accuracy', within=['mode', 'method'],
                  subject='subject', detailed=True)
print("双因素重复测量方差分析结果:")
print(aov)

# 方法1：从ANOVA结果中获取偏eta平方
print("效果量 (偏η²):")
print(aov[['Source', 'ng2']])

# 方法2：手动计算效果量（Cohen's d）


def calculate_cohens_d(data, dv, within_factors, subject):
    """计算配对样本的Cohen's d"""
    effect_sizes = {}

    for factor in within_factors:
        levels = data[factor].unique()
        for i in range(len(levels)):
            for j in range(i+1, len(levels)):
                # 获取配对数据
                subset1 = data[data[factor] == levels[i]]
                subset2 = data[data[factor] == levels[j]]

                # 确保按subject匹配
                merged = pd.merge(subset1, subset2,
                                  on='subject', suffixes=('_1', '_2'))

                # 计算配对差异
                diff = merged[f'{dv}_1'] - merged[f'{dv}_2']

                # 计算Cohen's d
                d = np.mean(diff) / np.std(diff, ddof=1)

                key = f"{factor}: {levels[i]} vs {levels[j]}"
                effect_sizes[key] = d

    return effect_sizes


# 计算效果量
effect_sizes = calculate_cohens_d(
    df, 'accuracy', ['mode', 'method'], 'subject')
print("\nCohen's d 效果量:")
for key, value in effect_sizes.items():
    print(f"{key}: {value:.3f}")

# %% ---- 2025-11-26 ------------------------
# Pending


# %%

# %%
