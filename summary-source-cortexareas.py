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
import statsmodels.api as sm

from scipy import stats
from itertools import product
from util.easy_import import *

sns.set_theme(context='paper', style='ticks', font_scale=2)

# %%
CACHE_DIR = Path(f'./data/cache/')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MODE = 'meg'  # 'meg' or 'eeg'

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


def add_top_left_notion(ax, notion='a'):
    ax.text(-0.1, 1.05, f'{notion})', transform=ax.transAxes,
            fontsize=12, va='bottom')
    return


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
dfs = []
for mode in ['meg', 'eeg']:
    cached_file = CACHE_DIR.joinpath(f'source_area_erds-{mode}.pkl')
    if cached_file.exists():
        obj = joblib.load(cached_file)
        df = obj['df']
        times = obj['times']
    else:
        array = []
        for subject, band, evt in tqdm(product([f'S{i:02d}' for i in range(1, 11)], ['alpha', 'beta'], ['1', '2', '3', '4', '5']), total=10*2*5):
            stc = read_compute_stc(subject, band, mode=mode, evt=evt)
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
        df = pd.DataFrame(array)
        joblib.dump({'df': df, 'times': times}, cached_file)

    df['mode'] = mode
    dfs.append(df)

large_table = pd.concat(dfs)

# large_table = large_table[large_table['area'].isin(['central', 'parietal'])]
large_table = large_table[large_table['area'].isin(['central'])]
display(times)
display(large_table)

# %%
group = large_table.groupby(['band', 'evt', 'area', 'sub_area', 'mode'])

a = group.agg(lambda e: np.stack(e, axis=0)).reset_index()
a['mean_erd'] = a['mean_erd'].map(lambda e: np.mean(e, axis=0))
a = a[['band', 'evt', 'area', 'sub_area', 'mode', 'mean_erd']]
b = []
for i, t in tqdm(enumerate(times), total=len(times)):
    c = a.copy()
    c['mean_erd'] = c['mean_erd'].map(lambda e: e[i])
    c['t'] = t
    b.append(c)
b = pd.concat(b)

display(b)

# %%
# g = sns.FacetGrid(b, col="band", row="mode", aspect=1)
# g.map_dataframe(sns.lineplot, x='t', y='mean_erd', hue='evt')
# g.add_legend()
# plt.show()

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
df_raw = df.copy()
display(df)

# %%
df_pivot = df.pivot_table(index=['evt', 'area', 'subject', 'sub_area', 't', 'band'],  # 保持其他标识列
                          columns='mode',
                          values='erd')

# 重置索引
df_pivot = df_pivot.reset_index()
df_pivot['evt'] = df_pivot['evt'].map(lambda e: TASK_TABLE[e])
display(df_pivot)

df_pivot.to_csv(CACHE_DIR.joinpath('events-scatter-meg-eeg-alpha-beta.csv'))
# %%

g = sns.lmplot(data=df_pivot, x='eeg', y='meg',
               hue='band', col='evt',  # 按事件分面
               markers='.',
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
g.set_titles(col_template="{col_name}", fontweight='bold')
plt.show()

# %%
# 对每个组进行线性回归分析，重点关注斜率检验
regression_results = []
for evt in df_pivot['evt'].unique():
    for band in df_pivot['band'].unique():
        subset = df_pivot[(df_pivot['evt'] == evt) &
                          (df_pivot['band'] == band)]
        clean_subset = subset.dropna(subset=['eeg', 'meg'])

        if len(clean_subset) > 2:
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_subset['eeg'], clean_subset['meg']
            )

            # 计算斜率的t统计量和p值（双边检验）
            n = len(clean_subset)
            df = n - 2  # 自由度

            # t统计量：斜率/标准误
            t_stat = slope / std_err

            # 斜率为零的双边t检验p值
            p_slope_zero = 2 * stats.t.sf(np.abs(t_stat), df)

            # 计算斜率的95%置信区间
            t_critical = stats.t.ppf(0.975, df)  # 95%置信度的t临界值
            ci_low = slope - t_critical * std_err
            ci_high = slope + t_critical * std_err

            regression_results.append({
                'Event': evt,
                'Band': band,
                'n': n,
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err,
                't_stat': t_stat,
                'p_slope': p_slope_zero,  # 斜率显著不为零的p值
                'ci_low': ci_low,
                'ci_high': ci_high,
                'r_value': r_value,
                'r_squared': r_value**2,
                'p_linregress': p_value,  # linregress返回的p值（与斜率检验等价）
                'significant': p_slope_zero < 0.05,
                'significance': '*' if p_slope_zero < 0.05 else 'ns'
            })
        else:
            print(f"警告: {evt} - {band} 数据不足，无法进行回归分析")

# 创建回归结果DataFrame
regression_df = pd.DataFrame(regression_results)

# 格式化显示


def format_p_value(p):
    if p < 0.0001:
        return "<0.0001"
    elif p < 0.001:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"


# 打印详细的斜率检验结果
print("=" * 90)
print("EEG-MEG 斜率检验结果（检验斜率是否显著不为零）")
print("=" * 90)

for evt in df_pivot['evt'].unique():
    print(f"\n事件: {evt}")
    print("-" * 50)

    evt_results = regression_df[regression_df['Event'] == evt]

    for _, row in evt_results.iterrows():
        p_formatted = format_p_value(row['p_slope'])

        print(f"{row['Band']}:")
        print(f"  斜率 = {row['slope']:.4f} ± {row['std_err']:.4f}")
        print(
            f"  t({row['n']-2}) = {row['t_stat']:.3f}, p = {p_formatted} {row['significance']}")
        print(f"  95% CI = [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        print(f"  截距 = {row['intercept']:.4f}, R² = {row['r_squared']:.3f}")
        print()

# 可选：创建汇总表格
print("\n" + "=" * 90)
print("斜率检验汇总")
print("=" * 90)

summary_table = regression_df[['Event', 'Band', 'n', 'slope', 'std_err',
                               't_stat', 'p_slope', 'significant', 'r_squared']].copy()
summary_table['p_slope'] = summary_table['p_slope'].apply(format_p_value)
print(summary_table.to_string(index=False))

# 保存结果
regression_df.to_csv(CACHE_DIR.joinpath('slope_test_results.csv'), index=False)

# 可选：进行多重比较校正
print("\n" + "=" * 90)
print("多重比较校正（Bonferroni校正）")
print("=" * 90)

# 获取所有p值
p_values = regression_df['p_slope'].values
num_tests = len(p_values)

# Bonferroni校正
alpha = 0.05
bonferroni_corrected = alpha / num_tests

print(f"总检验次数: {num_tests}")
print(f"原始α水平: {alpha}")
print(f"Bonferroni校正后α水平: {bonferroni_corrected:.6f}")

# 应用校正
regression_df['p_bonferroni'] = np.minimum(p_values * num_tests, 1.0)
regression_df['sig_bonferroni'] = regression_df['p_slope'] < bonferroni_corrected

for _, row in regression_df.iterrows():
    sig_original = "显著" if row['p_slope'] < 0.05 else "不显著"
    sig_bonferroni = "显著" if row['sig_bonferroni'] else "不显著"

    print(f"{row['Event']} - {row['Band']}: "
          f"原始p={format_p_value(row['p_slope'])}({sig_original}), "
          f"Bonferroni p={format_p_value(row['p_bonferroni'])}({sig_bonferroni})")

# %%
# Statistical analysis
# 对每个组进行线性回归分析
regression_results = []
for evt in df_pivot['evt'].unique():
    for band in df_pivot['band'].unique():
        subset = df_pivot[(df_pivot['evt'] == evt) &
                          (df_pivot['band'] == band)]
        clean_subset = subset.dropna(subset=['eeg', 'meg'])

        if len(clean_subset) > 2:
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_subset['eeg'], clean_subset['meg']
            )

            regression_results.append({
                'Event': evt,
                'Band': band,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            })

# 创建回归结果DataFrame
regression_df = pd.DataFrame(regression_results)
regression_df.to_csv(CACHE_DIR.joinpath('regression_results.csv'), index=False)
print(regression_df)


# %%

g = sns.lmplot(data=df_pivot, x='eeg', y='meg',
               hue='band', col='t',  # 按时间分面
               markers='.',
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
g.set_titles(col_template="t = {col_name}s", fontweight='bold')
plt.show()

df_pivot1 = df_pivot.copy()
df_pivot1['evt'] = df_pivot['evt'].map(
    lambda e: 'Rest' if e == 'Rest' else 'Task')

g = sns.lmplot(data=df_pivot1, x='eeg', y='meg',
               hue='band', col='evt',  # 按事件分面
               markers='.',
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
g.set_titles(col_template="{col_name}", fontweight='bold')
plt.show()

# %%
_df = df_pivot.query('evt != "Rest"')
g = sns.lmplot(data=_df, x='eeg', y='meg', hue='band',
               col='t',
               row='sub_area',
               markers='.',
               scatter_kws={'alpha': 0.5}, height=5, aspect=1, ci=95)  # 置信区间
plt.show()


# %%
df = df_raw

# 首先将数据转换为宽格式
df_pivot = df.pivot_table(index=['evt', 'area', 'subject', 'sub_area', 't', 'mode'],  # 保持其他标识列
                          columns='band',
                          values='erd')

# 重置索引
df_pivot = df_pivot.reset_index()
df_pivot['evt'] = df_pivot['evt'].map(lambda e: TASK_TABLE[e])
display(df_pivot)


# %%
sns.set_theme(context='paper', style='ticks', font_scale=2)

g = sns.lmplot(data=df_pivot, x='alpha', y='beta',
               hue='mode', col='evt',  # 按事件分面
               markers='.',
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
g.set_titles(col_template="{col_name}", fontweight='bold')
plt.show()

g = sns.lmplot(data=df_pivot, x='alpha', y='beta',
               hue='mode', col='t',  # 按时间分面
               markers='.',
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
g.set_titles(col_template="t = {col_name}s", fontweight='bold')
plt.show()

# %%
array = []
for evt in df_pivot['evt'].unique():
    for t in df_pivot['t'].unique():
        for mode in df_pivot['mode'].unique():
            if evt == 'Rest':
                continue
            # [df_pivot['evt'] == evt]
            df_evt = df_pivot.query(f'evt=="{evt}" & t=={t} & mode=="{mode}"')
            model = sm.OLS(df_evt['beta'], sm.add_constant(
                df_evt['alpha'])).fit()

            # 获取关心的四个核心指标
            R2 = model.rsquared
            alpha_coef = model.params[1]
            alpha_pvalue = model.pvalues[1]

            # 残差正态性检验 (Omnibus test)
            residuals = model.resid
            omnibus_stat, omnibus_pvalue = stats.normaltest(residuals)

            # 自相关检验 (Durbin-Watson)
            durbin_watson = sm.stats.stattools.durbin_watson(residuals)

            array.append(
                (evt, mode, t, R2, alpha_coef, omnibus_pvalue)
            )

df = pd.DataFrame(array, columns=[
                  'evt', 'mode', 't', 'R2', 'alpha_coef', 'omnibus_p'])
display(df)

sns.barplot(df, x='t', y='R2', hue='mode', legend=None)
plt.show()

# %%
array = []
for evt in df_pivot['evt'].unique():
    for t in df_pivot['t'].unique():
        for mode in df_pivot['mode'].unique():
            # [df_pivot['evt'] == evt]
            df_evt = df_pivot.query(f'evt=="{evt}" & t=={t} & mode=="{mode}"')
            model = sm.OLS(df_evt['beta'], sm.add_constant(
                df_evt['alpha'])).fit()
            print(
                f'{evt=}, {mode=}, R²={model.rsquared=:.4f}, p-value={model.pvalues[1]=:.4e}')

            # 获取关心的四个核心指标
            R2 = model.rsquared
            alpha_coef = model.params[1]
            alpha_pvalue = model.pvalues[1]

            # 残差正态性检验 (Omnibus test)
            residuals = model.resid
            omnibus_stat, omnibus_pvalue = stats.normaltest(residuals)

            # 自相关检验 (Durbin-Watson)
            durbin_watson = sm.stats.stattools.durbin_watson(residuals)

            array.append(
                (evt, mode, t, R2, alpha_coef, omnibus_pvalue)
            )

            print(model.summary())

df = pd.DataFrame(array, columns=[
                  'evt', 'mode', 't', 'R2', 'alpha_coef', 'omnibus_p'])
df.to_csv(CACHE_DIR.joinpath('events-regression-r2.csv'))
display(df)

sns.barplot(df, x='evt', y='R2', hue='mode', legend=None)
plt.show()


# %%
# exit(0)

# %%
g = sns.lmplot(data=df_pivot, x='alpha', y='beta',
               hue='area', col='t', row='evt',  # 按事件和时间分面
               scatter_kws={'alpha': 0.5},
               height=5, aspect=1,
               ci=95)  # 置信区间
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
