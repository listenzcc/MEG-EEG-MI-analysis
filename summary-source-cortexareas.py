"""
File: summary-source-cortexareas.py
Author: Chuncheng Zhang
Date: 2025-11-21
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Working on the source analysis results.
    Parse the ERDs for every cortex area of interest.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-11-21 ------------------------
# Requirements and constants
import seaborn as sns
import seaborn.objects as so
from itertools import product
from util.easy_import import *

# %%
CACHE_DIR = Path(f'./data/cache/')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# %%
TASK_TABLE = {
    '1': 'Hand',
    '2': 'Wrist',
    '3': 'Elbow',
    '4': 'Shoulder',
    '5': 'Rest'
}

# %% ---- 2025-11-21 ------------------------
# Function and class


def load_labels(parc='aparc_sub'):
    labels_parc = mne.read_labels_from_annot(
        'fsaverage', parc=parc, subjects_dir=mne.datasets.fetch_fsaverage().parent
    )
    df = pd.DataFrame(
        [(e.name, e) for e in labels_parc], columns=["name", "label"]
    )

    assert len(df['name'].unique()) == len(df), "Duplicate label names found!"

    df.index = df['name']
    return df


def find_stc(subject, band, mode, evt):
    folder = Path(f'./data/tfr-stc-{band}/')
    subjects = [e.name for e in folder.iterdir()]
    subject = [e for e in subjects if e.startswith(subject)][0]
    fpath = folder.joinpath(f'{subject}/{mode}-evt{evt}.stc')
    return fpath


def read_compute_stc(subject, band, mode, evt, tmin=-1, tmax=2):
    fpath = find_stc(subject, band, mode, evt)

    stc = mne.read_source_estimate(fpath)
    stc.crop(tmin=tmin, tmax=tmax)

    # Ratio by the baseline (-1, 0)
    data = stc.data
    times = stc.times
    base_data = np.mean(data[:, times < 0], axis=1)
    extended_base_data = np.stack([base_data for e in times]).T

    # Convert into dB
    data = 10 * np.log10(data/extended_base_data)
    stc.data = data

    return stc


class ROI:
    names = {
        'central': [f'postcentral_{i}' for i in [3, 4, 5, 6, 7, 8, 9]] + [f'precentral_{i}' for i in [5, 6, 7, 8]],
        'parietal': [f'superiorparietal_{i}' for i in [11, 12]] + [f'inferiorparietal_{i}' for i in [1, 4, 3, 8]],
        'occipital': [f'lateraloccipital_{i}' for i in [2, 3, 6, 7, 8, 9]]
    }
    colors = {
        'central': 'cyan',
        'parietal': 'green',
        'occipital': 'blue'
    }


# %% ---- 2025-11-21 ------------------------
# Play ground
label_df = load_labels(parc='aparc_sub')
display(label_df)

# %%
cached_file = CACHE_DIR.joinpath('source_area_erds.pkl')
if cached_file.exists():
    obj = joblib.load(cached_file)
    large_table = obj['df']
    times = obj['times']
else:
    array = []
    for subject, band, evt in tqdm(product([f'S{i:02d}' for i in range(1, 11)], ['alpha', 'beta'], ['1', '2', '3', '4', '5']), total=10*2*5):
        stc = read_compute_stc(subject, band, mode='meg', evt=evt)
        stc.crop(tmin=0, tmax=1.5)
        for area, sub_areas in ROI.names.items():
            for sub_area in sub_areas:
                label = label_df.loc[f'{sub_area}-lh', "label"]
                # stc.data shape is (n_vertices, n_times)
                array.append({
                    'subject': subject,
                    'band': band,
                    'evt': evt,
                    'area': area,
                    'sub_area': sub_area,
                    'mean_erd': np.min(stc.in_label(label).data, axis=0)
                })
    times = stc.times
    large_table = pd.DataFrame(array)
    joblib.dump({'df': large_table, 'times': times}, cached_file)

large_table = large_table[large_table['area'].isin(['central', 'parietal'])]
display(times)
display(large_table)

# %%
dfs = []
ts = [0.2, 0.4, 0.6, 0.8, 1.0]
for t in tqdm(ts):
    df = large_table.copy()
    time_idx = np.argmin(np.abs(times - t))
    df['erd'] = df['mean_erd'].map(lambda e: e[time_idx])
    df['t'] = t
    dfs.append(df)
df = pd.concat(dfs)
display(df)

# %%

# 首先将数据转换为宽格式
df_pivot = df.pivot_table(index=['evt', 'area', 'subject', 'sub_area', 't'],  # 保持其他标识列
                          columns='band',
                          values='erd')

# 重置索引
df_pivot = df_pivot.reset_index()
display(df_pivot)

# %%
sns.set_theme()

g = sns.lmplot(data=df_pivot, x='alpha', y='beta',
               hue='area', col='evt',  # 按事件分面
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
plt.show()

g = sns.lmplot(data=df_pivot, x='alpha', y='beta',
               hue='area', col='t',  # 按时间分面
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
plt.show()

# %%
ax = sns.lmplot(df_pivot, x='alpha', y='beta', hue='area')
plt.show()


# %%
# Compute correlation between alpha and beta bands for every area and event


array = []
for subject, evt, area in product(large_table['subject'].unique(),
                                  large_table['evt'].unique(),
                                  large_table['area'].unique()):
    df_sub = large_table[
        (large_table['subject'] == subject) &
        (large_table['evt'] == evt) &
        (large_table['area'] == area)
    ]
    alpha_erd = np.mean(
        df_sub[df_sub['band'] == 'alpha']['mean_erd'].values, axis=0)
    beta_erd = np.mean(
        df_sub[df_sub['band'] == 'beta']['mean_erd'].values, axis=0)
    corr = np.corrcoef(alpha_erd, beta_erd)[0, 1]
    corr = np.abs(corr)
    array.append({
        'subject': subject,
        'evt': evt,
        'area': area,
        'alpha_beta_correlation': corr})
df = pd.DataFrame(array)
display(df)

ax = sns.boxplot(df, x='evt', y='alpha_beta_correlation',
                 hue='area', showfliers=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.show()


# %%


# %% ---- 2025-11-21 ------------------------
# Pending
dfs = []
ts = [0.2, 0.4, 0.6, 0.8, 1.0]
for t in tqdm(ts):
    df = large_table.copy()
    time_idx = np.argmin(np.abs(times - t))
    df['erd'] = df['mean_erd'].map(lambda e: e[time_idx])
    df['t'] = t
    dfs.append(df)

df = pd.concat(dfs)
df['evt'] = df['evt'].map(lambda e: TASK_TABLE[e])
display(df)

# %%
g = sns.FacetGrid(df, col='t')
g.map_dataframe(sns.lineplot, x='evt', y='erd', hue='area', style='band')
g.add_legend()
plt.show()

# %% ---- 2025-11-21 ------------------------
# Pending
g = sns.FacetGrid(df, col='t', row='area')
g.map_dataframe(sns.lineplot, x='evt', y='erd',
                hue='sub_area', style='band', errorbar=None)
g.add_legend()
plt.show()

# %%
