# %%
import numpy as np
import pandas as pd

from scipy import stats
from IPython.display import display
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# %%
table = pd.read_csv(
    './data/MVPA.FBCSP.compare.MEG-EEG.vote.results/df_compare.csv',
    index_col=0
)
table

# %%
# 1. Select MEG and EEG records
#    chHemi = B, chMark = C, P, F

marks = ['C', 'P', 'F']
modalities = ['MEG', 'EEG']

df_compare = table[
    (table['modality'].isin(modalities)) &
    (table['chHemi'] == 'B') &
    (table['chMark'].isin(marks))
].copy()

# Keep only subjects with all six conditions
# (MEG/EEG x C/P/F)
wide_check = df_compare.pivot(
    index='subject',
    columns=['modality', 'chMark'],
    values='acc'
)

required_columns = pd.MultiIndex.from_product(
    [modalities, marks],
    names=['modality', 'chMark']
)

wide_check = wide_check.reindex(columns=required_columns)

valid_subjects = wide_check.dropna().index

df_compare = df_compare[
    df_compare['subject'].isin(valid_subjects)
].copy()

print(f'Number of paired subjects: {len(valid_subjects)}')
print(df_compare.groupby(['modality', 'chMark']).size())

# %%
# 2. Two-way ANOVA
#    Factors: modality x chMark
#    Dependent variable: acc

model = ols(
    'acc ~ C(modality) * C(chMark)',
    data=df_compare
).fit()

anova_results = anova_lm(model, typ=2)

print('\nTwo-way ANOVA (MEG vs EEG, chHemi=B)')
display(anova_results.round(5))
anova_results.to_csv('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results/anova_meg_eeg.csv')

# %%
# 3. Paired t-tests: MEG vs EEG
#    Separate comparisons for C, P, F

wide = df_compare.pivot(
    index='subject',
    columns=['modality', 'chMark'],
    values='acc'
)

results = []

for mark in marks:

    paired = wide[[('MEG', mark), ('EEG', mark)]].dropna()

    meg = paired[('MEG', mark)]
    eeg = paired[('EEG', mark)]

    t_stat, p_value = stats.ttest_rel(meg, eeg)

    diff = meg - eeg

    results.append({
        'chMark': mark,
        'N': len(paired),
        'MEG_mean': meg.mean(),
        'EEG_mean': eeg.mean(),
        'Mean_Diff': diff.mean(),
        't': t_stat,
        'df': len(paired) - 1,
        'p': p_value
    })

ttest_results = pd.DataFrame(results)

# Multiple comparison correction (fdr)
ttest_results['p_fdr'] = multipletests(
    ttest_results['p'],
    method='fdr_bh', # 'holm'
)[1]

ttest_results['Significant'] = (
    ttest_results['p_fdr'] < 0.05
)

print('\nPaired t-tests (MEG vs EEG, chHemi=B)')
display(ttest_results.round(5))
ttest_results.to_csv('./data/MVPA.FBCSP.compare.MEG-EEG.vote.results/ttest_meg_eeg.csv')

# %%